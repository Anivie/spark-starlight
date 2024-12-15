use inference_yolo_detect::YoloDetectResult;

pub mod inference_yolo_seg;
pub mod inference_yolo_detect;


impl YoloDetectResult {
    // 计算两个检测框的IoU
    pub fn iou(&self, other: &YoloDetectResult) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        let intersection_width = (x2 - x1).max(0.0);
        let intersection_height = (y2 - y1).max(0.0);
        let intersection_area = intersection_width * intersection_height;

        let self_area = self.width * self.height;
        let other_area = other.width * other.height;

        intersection_area / (self_area + other_area - intersection_area)
    }
}

pub trait NMSImplement {
    fn non_maximum_suppression(
        self,
        iou_threshold: f32,
        score_threshold: f32,
        class_index: usize,
    ) -> Vec<YoloDetectResult>;
}

impl NMSImplement for Vec<YoloDetectResult> {
    fn non_maximum_suppression(
        self,
        iou_threshold: f32,
        score_threshold: f32,
        class_index: usize, // 指定分类的索引
    ) -> Vec<YoloDetectResult> {
        let mut filtered_detections: Vec<YoloDetectResult> = self
            .into_iter()
            .filter(|d| d.score[class_index] >= score_threshold)
            .collect();

        // 根据分类得分排序（降序）
        filtered_detections.sort_by(|a, b| {
            b.score[class_index]
                .partial_cmp(&a.score[class_index])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut result: Vec<YoloDetectResult> = Vec::new();

        while !filtered_detections.is_empty() {
            // 选择得分最高的框并移除
            let best = filtered_detections.remove(0);
            result.push(best.clone());

            // 过滤掉与当前框的IoU超过阈值的框
            filtered_detections.retain(|d| best.iou(d) < iou_threshold);
        }

        result
    }
}