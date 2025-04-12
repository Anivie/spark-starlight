use crate::detect::analysis::{Describer, DescriberDispatcher};
use crate::detect::property::analyse_result::RoadAnalysisData;

pub struct CompositeDescriber {
    describers: Vec<DescriberDispatcher>,
}

impl CompositeDescriber {
    pub fn new() -> Self {
        Self {
            describers: DescriberDispatcher::all(),
        }
    }

    /// Generates the full description by combining outputs from individual describers.
    pub async fn describe(&self, data: &RoadAnalysisData<'_>, object_type_name: &str) -> String {
        let mut parts: Vec<String> = Vec::new();

        let back = self.describers.describe(data, object_type_name).await;
        if let Some(desc) = back {
            parts.push(desc);
        }

        if parts.is_empty() {
            // Handle the case where absolutely nothing could be described
            // This might happen if core analysis returns None or if all describers return None.
            // Check the RoadStartDescriber's initial message if it was generated.
            // A more robust check could involve looking at the initial data more directly.
            if data.center_lines.is_empty() && data.detect_results.is_empty() {
                return format!("No {} or related objects detected.", object_type_name);
            } else if data.center_lines.is_empty() {
                // If objects were detected but no road, DetectedObjectDescriber should have run.
                // If it didn't produce output for some reason, provide a fallback.
                return format!(
                    "Could not clearly identify the {}. No specific objects described.",
                    object_type_name
                );
            } else {
                // Has centerline, but maybe no shape, no obstacles, no objects? Unlikely but possible.
                return format!(
                    "Analysis complete for {}. Path appears clear.",
                    object_type_name
                );
            }
        }

        // Combine the parts into a single sentence-like structure.
        // Capitalize the first part, add periods, and join with ". ".
        if let Some(first_part) = parts.first_mut() {
            let mut c = first_part.chars();
            if let Some(f) = c.next() {
                *first_part = f.to_uppercase().collect::<String>() + c.as_str();
            }
        }

        let mut final_description = parts.join(". ");
        // Ensure the final string ends with a period.
        if !final_description.ends_with('.') {
            final_description.push('.');
        }

        final_description
    }
}

// Optional: Implement the Describer trait for CompositeDescriber itself
// if you want to treat the composition as a single Describer unit elsewhere.
impl Describer for CompositeDescriber {
    async fn describe(
        &self,
        data: &RoadAnalysisData<'_>,
        object_type_name: &str,
    ) -> Option<String> {
        // The composite always aims to produce *some* description, so wrap in Some()
        // unless it's truly empty after joining.
        let desc = self.describe(data, object_type_name).await;
        if desc.is_empty() || desc == "." {
            // Check for empty or just a period
            None
        } else {
            Some(desc)
        }
    }
}
