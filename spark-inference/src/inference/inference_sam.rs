use crate::engine::inference_engine::OnnxSession;
use crate::utils::graph::Point;
use crate::INFERENCE_CUDA;
use anyhow::Result;
use bitvec::prelude::*;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use ndarray::{array, Array1, Array2, Array3, ArrayViewD};
use ort::{inputs, AllocationDevice, AllocatorType, MemoryInfo, MemoryType, SessionInputValue, SessionOutputs, TensorRefMut};
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::ops::Deref;
use std::sync::LazyLock;

pub trait SamModelInference {
    fn inference_sam(&self, points: Vec<Point<u32>>, image: &mut Image) -> Result<BitVec>;
}

pub struct SAM2InferenceSession {
    image_encoder: OnnxSession,
    image_decoder: OnnxSession,
    memory_attention: OnnxSession,
    memory_encoder: OnnxSession,
}

impl SAM2InferenceSession {
    pub fn new(
        image_encoder: OnnxSession, image_decoder: OnnxSession,
        memory_attention: OnnxSession, memory_encoder: OnnxSession
    ) -> Self {
        Self {
            image_encoder,
            image_decoder,
            memory_attention,
            memory_encoder,
        }
    }
}

impl SamModelInference for SAM2InferenceSession {
    fn inference_sam(&self, points: Vec<Point<u32>>, image: &mut Image) -> Result<BitVec> {
        let filter = {
            let mut filter = AVFilter::new(image.pixel_format()?, image.get_size())?;
            filter.add_context("scale", "1024:1024")?;
            filter.add_context("format", "rgb24")?;
            filter.lock()?
        };

        let image_size = image.get_size();
        image.apply_filter(&filter)?;

        // let tensor = image.extra_standard_image_to_tensor()?;
        let encoder_output = self.inference_image_encoder(image.raw_data()?.deref())?;

        let decoder_output = self.inference_image_decoder(
            encoder_output["vision_feats"].try_extract_tensor::<f32>()?,
            encoder_output["high_res_feat0"].try_extract_tensor::<f32>()?,
            encoder_output["high_res_feat1"].try_extract_tensor::<f32>()?,
            points,
            image_size
        )?;

        let pred_mask = decoder_output["pred_mask"].try_extract_tensor::<f32>()?;

        /*let memory_encoder_output = self.inference_memory_encoder(
            decoder_output["mask_for_mem"].try_extract_tensor::<f32>()?.view(),
            encoder_output["pix_feat"].try_extract_tensor::<f32>()?.view()
        )?;*/

        let mut back = BitVec::with_capacity((image.get_width() * image.get_height()) as usize);

        println!("shape: {:?}", pred_mask.shape());
        pred_mask.iter().for_each(|x| {
            back.push(*x > 0f32);
        });

        Ok(back)
    }
}

impl SAM2InferenceSession {
    fn inference_image_encoder(&self, image: &Vec<u8>) -> Result<SessionOutputs> {
        static MEAN: LazyLock<CudaSlice<f32>> = LazyLock::new(|| {
            INFERENCE_CUDA.htod_sync_copy(&[0.485, 0.456, 0.406]).unwrap()
        });
        static STD: LazyLock<CudaSlice<f32>> = LazyLock::new(|| {
            INFERENCE_CUDA.htod_sync_copy(&[0.229, 0.224, 0.225]).unwrap()
        });

        let buffer = INFERENCE_CUDA.htod_sync_copy(image.as_slice())?;
        let tensor: TensorRefMut<'_, f32> = unsafe {
            let mut out = INFERENCE_CUDA.alloc::<f32>(image.len())?;
            let cfg = LaunchConfig::for_num_elems((image.len() / 3) as u32);

            INFERENCE_CUDA.normalise_pixel_mean().launch(cfg, (
                &mut out, &buffer,
                MEAN.deref(), STD.deref(),
                image.len(),
            ))?;

            TensorRefMut::from_raw(
                MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
                (*out.device_ptr() as usize as *mut ()).cast(),
                vec![1, 3, 1024, 1024],
            )?
        };

        INFERENCE_CUDA.synchronize()?;
        Ok(self.image_encoder.run(vec![("image", SessionInputValue::from(tensor))])?)
    }

    fn inference_memory_encoder(
        &self,
        mask_for_mem: ArrayViewD<f32>,
        pix_feat: ArrayViewD<f32>,
    ) -> Result<SessionOutputs> {
        let result = self.memory_encoder.run(inputs![
            "mask_for_mem" => mask_for_mem,
            "pix_feat" => pix_feat,
        ]?)?;

        Ok(result)
    }

    fn inference_image_decoder(
        &self,

        feats: ArrayViewD<f32>,
        feat0: ArrayViewD<f32>,
        feat1: ArrayViewD<f32>,

        points: Vec<Point<u32>>,
        image_size: (i32, i32),
    ) -> Result<SessionOutputs> {
        let point_labels = Array2::from_shape_vec(
            (1, points.len()), vec![1_f32; points.len()]
        )?;

        let points = points
            .iter()
            .map(|point| array![
                1024f32 * (point.x as f32 / image_size.0 as f32),
                1024f32 * (point.y as f32 / image_size.1 as f32),
            ])
            .collect::<Vec<Array1<f32>>>();

        let points = Array3::from_shape_vec(
            (1, points.len(), 2), points.into_iter().flatten().collect()
        )?;

        let image_size = array![image_size.1 as i64, image_size.0 as i64];

        let result = self.image_decoder.run(inputs![
            "point_coords" => points,
            "point_labels" => point_labels,
            "frame_size" => image_size,

            "image_embed" => feats,
            "high_res_feats_0" => feat0,
            "high_res_feats_1" => feat1,
        ]?)?;

        Ok(result)
    }
}