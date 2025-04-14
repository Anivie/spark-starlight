use std::io::Read;

// #[test]
pub fn debug() {
    use spark_inference::inference::sam::image_inference::{
        SAMImageInferenceSession, SamImageInference,
    };
    use spark_inference::inference::yolo::inference_yolo_detect::{
        YoloDetectInference, YoloDetectSession,
    };
    use spark_inference::inference::yolo::NMSImplement;
    use spark_inference::utils::graph::SamPrompt;
    use spark_inference::utils::masks::ApplyMask;
    use spark_media::filter::filter::AVFilter;
    use spark_media::{Image, RGB};
    let tst: fn() -> anyhow::Result<()> = || {
        let yolo = YoloDetectSession::new("./data/model")?;
        let sam2 = SAMImageInferenceSession::new("./data/model/other5")?;

        let path = "./data/image/rt.jpeg";
        // let mut image = Image::open_file(path)?;
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        // let image = Image::open_file(path)?;
        let mut image = Image::from_bytes(buffer.as_slice())?;

        let sam_image = image.clone();

        let results = yolo.inference_yolo(image.clone(), 0.25)?;
        println!("results: {:?}", results.len());

        let result_highway = results
            .clone()
            .into_iter()
            .filter(|result| result.score[0] >= 0.8)
            .collect::<Vec<_>>();
        let result_sidewalk = results
            .into_iter()
            .filter(|result| result.score[1] >= 0.4)
            .collect::<Vec<_>>();
        let result_highway = result_highway.non_maximum_suppression(0.5, 0.35, 0);
        let result_sidewalk = result_sidewalk.non_maximum_suppression(0.5, 0.25, 1);
        println!("highway: {:?}", result_highway);
        println!("sidewalk: {:?}", result_sidewalk);

        let (image_w, image_h) = image.get_size();
        let mut filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
            .add_context("scale", "1024:1024")?
            .add_context("format", "rgb24")?;

        for mask in result_highway.iter() {
            let string1 = format!(
                "x=({x}-{width}/2):y=({y}-{height}/2):w={width}:h={height}:color=red@1.0:t=6",
                x = mask.x / image_w as f32 * 1024.0,
                y = mask.y / image_h as f32 * 1024.0,
                width = mask.width / image_w as f32 * 1024.0,
                height = mask.height / image_h as f32 * 1024.0,
            );
            filter = filter.add_context("drawbox", string1.as_str())?
        }
        for x in result_sidewalk.iter() {
            let string = format!(
                "x=({x}-{width}/2):y=({y}-{height}/2):w={width}:h={height}:color=blue@1.0:t=6",
                x = x.x / image_w as f32 * 1024.0,
                y = x.y / image_h as f32 * 1024.0,
                width = x.width / image_w as f32 * 1024.0,
                height = x.height / image_h as f32 * 1024.0,
            );
            filter = filter.add_context("drawbox", string.as_str())?
        }
        image.apply_filter(&filter.build()?)?;

        let result_highway = result_highway
            .iter()
            .map(|yolo| {
                SamPrompt::both(
                    (
                        yolo.x - yolo.width / 2.0,
                        yolo.y - yolo.height / 2.0,
                        yolo.x + yolo.width / 2.0,
                        yolo.y + yolo.height / 2.0,
                    ),
                    (yolo.x, yolo.y),
                )
            })
            .collect::<Vec<_>>();

        let result_sidewalk = result_sidewalk
            .iter()
            .map(|yolo| {
                SamPrompt::both(
                    (
                        yolo.x - yolo.width / 2.0,
                        yolo.y - yolo.height / 2.0,
                        yolo.x + yolo.width / 2.0,
                        yolo.y + yolo.height / 2.0,
                    ),
                    (yolo.x, yolo.y),
                )
            })
            .collect::<Vec<_>>();

        let mask = sam2.inference_frame(
            sam_image,
            Some((1024, 1024)),
            vec![result_highway, result_sidewalk],
        )?;

        for x in &mask[0] {
            image.layering_mask(&x, RGB(75, 0, 0))?;
        }
        for x in &mask[1] {
            image.layering_mask(&x, RGB(0, 0, 75))?;
        }

        image.save_with_format("./data/out/e_out.png")?;
        Ok(())
    };

    tst().unwrap();
}
