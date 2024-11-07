use crate::engine::inference_engine::OnnxSession;
use crate::INFERENCE_CUDA;
use anyhow::Result;
use bitvec::prelude::*;
use cudarc::driver::{DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use ort::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType, SessionInputValue, SessionOutputs, TensorRefMut};
use rayon::prelude::*;
use spark_media::filter::filter::AVFilter;
use spark_media::Image;

pub trait SamModelInference {
    fn inference_sam(&self) -> Result<()>;
}

pub struct SAM2InferenceSession {
    image_encoder: OnnxSession,
    image_decoder: OnnxSession,
    memory_attention: OnnxSession,
    memory_encoder: OnnxSession,
}

impl SAM2InferenceSession {
    pub fn new(image_encoder: OnnxSession, image_decoder: OnnxSession, memory_attention: OnnxSession, memory_encoder: OnnxSession) -> Self {
        Self {
            image_encoder,
            image_decoder,
            memory_attention,
            memory_encoder,
        }
    }
}

impl SamModelInference for SAM2InferenceSession {
    fn inference_sam(&self) -> Result<()> {
        let mut image = Image::open_file("./data/image/a.png")?;

        let filter = {
            let mut filter = AVFilter::new(image.pixel_format()?, image.get_size())?;
            filter.add_context("scale", "1024:1024")?;
            filter.add_context("format", "rgb24")?;
            filter.lock()?
        };
        image.apply_filter(&filter)?;

        let encoder_output = self.inference_image_encoder(&image)?;





        Ok(())
    }
}

impl SAM2InferenceSession {
    fn inference_image_encoder(&self, image: &Image) -> Result<SessionOutputs> {
        let pixel_count = {
            let size = image.get_size();
            (size.0 * size.1 * 3) as u32
        };

        let cfg = LaunchConfig::for_num_elems(pixel_count);
        let out_buffer = unsafe {
            let buffer = INFERENCE_CUDA.htod_sync_copy(image.raw_data()?.as_slice())?;
            let mut out_buffer = INFERENCE_CUDA.alloc::<f32>(buffer.len())?;
            INFERENCE_CUDA.normalise_pixel().launch(cfg, (&mut out_buffer, &buffer, buffer.len()))?;
            INFERENCE_CUDA.synchronize()?;
            out_buffer
        };
        let tensor: TensorRefMut<'_, f32> = unsafe {
            TensorRefMut::from_raw(
                MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
                (*out_buffer.device_ptr() as usize as *mut ()).cast(),
                vec![1, 3, 1024, 1024],
            )?
        };
        Ok(self.image_encoder.run(vec![("image", SessionInputValue::from(tensor))])?)
    }
}