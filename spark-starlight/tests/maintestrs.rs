use bitvec::bitvec;
use bitvec::prelude::Lsb0;
use spark_inference::utils::masks::ApplyMask;
use spark_media::{Image, RGB};
use spark_media::image::decoder::size::ResizeImage;

#[test]
fn test() -> anyhow::Result<()> {
    let mut image = {
        let mut image = Image::open_file("/home/spark-starlight/data/image/a.png")?;
        image.resize_to((640, 640))?;
        image
    };
    let mask = {
        let mut mask = bitvec![usize, Lsb0;];
        for _ in 0..640 * 640 / 2 {
            mask.push(true);
        }
        for _ in 0..640 * 640 / 2 {
            mask.push(false);
        }
        mask
    };
    let mut new_image = image.clone();
    new_image.layering_mask(&mask, RGB(25, 25, 0))?;
    new_image.save("./data/out/test_lays.png")?;

    image.layering_mask(&mask, RGB(0, 25, 25))?;
    image.save("./data/out/test_lay.png")?;

    Ok(())
}
