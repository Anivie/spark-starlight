mod entity;

use cudarc::driver::{CudaDevice, DevicePtr};
use entity::box_point::Box;
use log::debug;
use ndarray::{s, Array2, Axis, CowArray, Dim, Ix, Ix2, Ix3};
use ort::{AllocationDevice, AllocatorType, CUDAExecutionProvider, MemoryInfo, MemoryType, Session, TensorRTExecutionProvider, TensorRefMut};
use rayon::prelude::*;
use spark_media::image_util::extract::ExtraToTensor;
use spark_media::Image;
use std::cmp::Ordering;
use std::sync::LazyLock;
use spark_ffmpeg::pixformat::AVPixelFormat;
use spark_ffmpeg::pixformat::AVPixelFormat::{AvPixFmtGray8, AvPixFmtRgb24};

const IMG_URL: &str = r#"./data/image/c.jpg"#;
const MODEL_URL: &str = r#"./data/model/best.onnx"#;

/*static SESSION: LazyLock<Session> = LazyLock::new(|| {
    // let provider = TensorRTExecutionProvider::default().build().error_on_failure();
    let provider = CUDAExecutionProvider::default().build().error_on_failure();
    ort::init()
        .with_execution_providers([provider])
        .commit()
        .unwrap();
    let model = Session::builder()
        .unwrap()
        .commit_from_file(MODEL_URL)
        .unwrap();
    model
});
*/
pub fn inference() -> anyhow::Result<()> {
    // let provider = TensorRTExecutionProvider::default().build().error_on_failure();
    let provider = CUDAExecutionProvider::default().build().error_on_failure();
    ort::init()
        .with_execution_providers([provider])
        .commit()?;
    let model = Session::builder()?.commit_from_file(MODEL_URL)?;

    debug!("Start reading image");
    let mut image = Image::open("/home/spark-starlight/data/image/b.png")?;
    debug!("Finish reading image");
    image.decode()?;
    debug!("Finish decode image");
    let frame = image.resize((640, 640), AvPixFmtRgb24)?;
    let tensor_raw = frame.extra_standard_image_to_tensor()?;
    debug!("Finish converting image to tensor");

    let tensor = {
        let device = CudaDevice::new(0)?;
        let device_data = device.htod_sync_copy(tensor_raw.as_slice())?;
        let tensor: TensorRefMut<'_, f32> = unsafe {
            TensorRefMut::from_raw(
                MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
                (*device_data.device_ptr() as usize as *mut ()).cast(),
                vec![1, 3, 640, 640],
            )?
        };
        tensor
    };

    debug!("Finish copying tensor to device");
    let outputs = model.run([tensor.into()])?;
    debug!("Finish running model");

    let output_first = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();
    let output_first = output_first.squeeze().into_dimensionality::<Ix2>()?;

    let output_second = outputs["output1"].try_extract_tensor::<f32>()?.into_owned();
    let output_second = output_second.squeeze().into_dimensionality::<Ix3>()?;
    let output_second = output_second.to_shape((32, 25600))?;

    debug!("output_first: {:?}", output_first.shape());
    debug!("output_second: {:?}", output_second.shape());

    let mask = output_first
        .axis_iter(Axis(0))
        .into_par_iter()
        .filter(|box_output| {
            box_output
                .slice(s![4 .. box_output.len() - 32])
                .iter()
                .any(|&score| score > 0.7)
        })
        .map(|box_output| {
            let score = box_output.slice(s![4 .. box_output.len() - 32]);
            let score = score.iter().max_by(|&a, &b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap_or(&0.0);
            (box_output, *score)
        })
        .map(|(box_output, score)| {
            let (x, y, w, h) = (box_output[0], box_output[1], box_output[2], box_output[3]);
            let x1 = x - w / 2.;
            let y1 = y - h / 2.;
            let x2 = x + w / 2.;
            let y2 = y + h / 2.;

            let feature_mask = box_output.slice(s![box_output.len() - 32 ..]);
            let feature_mask = feature_mask.to_shape((1, 32)).unwrap();

            let mask = {
                let mask = feature_mask.dot(&output_second);
                let mask = mask.to_shape((160, 160)).unwrap();
                let mask = bilinear_interpolate(&mask, (640, 640));
                let mut mask = sigmoid(mask.into_owned());
                let x1 = x1 as usize;
                let y1 = y1 as usize;
                let x2 = x2 as usize;
                let y2 = y2 as usize;

                mask
                    .indexed_iter_mut()
                    .par_bridge()
                    .for_each(|((y, x), value)| {
                        *value = if (y1 .. y2).contains(&y) && (x1 .. x2).contains(&x) && *value > 0.5 {
                            100.
                        }else {
                            0.
                        }
                    });
                mask
            };
            let boxed = Box { x1, y1, x2, y2 };

            (boxed, mask, score)
        })
        .collect::<Vec<_>>();

    for (index, (box_point, mask, score)) in mask.iter().enumerate() {
        let frame = image.resize((640, 640), AvPixFmtRgb24)?;
        let mut data = frame.get_raw_data(0);
        let mut image = Image::from_data((640, 640), AvPixFmtRgb24, 61)?;
        let packet = image.fill_data(data.as_mut_slice())?;
        packet.save(format!("/home/spark-starlight/data/out/mask_{}_iou_{}.png", index, score))?;
    }


    // image.fill_data()?;

/*    let mut image = DynamicImage::new_rgb8(640, 640);
    debug!("image size: {:?}", image.dimensions());

    for (index, (box_point, mask, score)) in mask.iter().enumerate() {
        mask
            .indexed_iter()
            .for_each(|((y, x), &value)| {
                if value == 255. {
                    let rgba = image.get_pixel(x as u32, y as u32).0;
                    image.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgba([
                            rgba[0],
                            if rgba[1] < 155 {
                                rgba[1] + 100
                            } else {
                                255
                            },
                            if rgba[2] < 155 {
                                rgba[2] + 100
                            } else {
                                255
                            },
                            255
                        ])
                    );
                }
            });
        image.save(format!("./data/out/mask_{}_iou_{}.png", index, score))?;
    }
*/
    Ok(())
}

fn bilinear_interpolate(input: &CowArray<f32, Dim<[Ix; 2]>>, new_shape: (usize, usize)) -> Array2<f32> {
    let (old_height, old_width) = input.dim();
    let (new_height, new_width) = new_shape;
    let mut output = Array2::<f32>::zeros((new_height, new_width));

    for i in 0..new_height {
        for j in 0..new_width {
            // Mapping new coordinates to old coordinates
            let x = (j as f32) / (new_width as f32) * (old_width as f32 - 1.0);
            let y = (i as f32) / (new_height as f32) * (old_height as f32 - 1.0);

            let x0 = x.floor() as usize;
            let x1 = x.ceil() as usize;
            let y0 = y.floor() as usize;
            let y1 = y.ceil() as usize;

            let p00 = input[[y0, x0]];
            let p01 = input[[y0, x1]];
            let p10 = input[[y1, x0]];
            let p11 = input[[y1, x1]];

            // Interpolation weights
            let dx = x - x0 as f32;
            let dy = y - y0 as f32;

            // Bilinear interpolation formula
            let interpolated_value =
                p00 * (1.0 - dx) * (1.0 - dy) +
                    p01 * dx * (1.0 - dy) +
                    p10 * (1.0 - dx) * dy +
                    p11 * dx * dy;

            output[[i, j]] = interpolated_value;
        }
    }

    output
}

fn sigmoid(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}