macro_rules! define_describer {
    ($($name: ident => $to: ident, )*) => {
        #[derive(Debug, Copy, Clone)]
        enum DescriberDispatcher {
            $(
                $name($to),
            )*
        }

        impl Describer for DescriberDispatcher {
            async fn describe(&self, data: &RoadAnalysisData<'_>, object_type_name: &str) -> Option<String> {
                match self {
                    $(
                        DescriberDispatcher::$name(describer) => describer.describe(data, object_type_name).await,
                    )*
                }
            }
        }

        impl Describer for Vec<DescriberDispatcher> {
            async fn describe(&self, data: &RoadAnalysisData<'_>, object_type_name: &str) -> Option<String> {
                let mut descriptions = Vec::new();
                let futures = self.iter().map(|describer| {
                    describer.describe(&data, &object_type_name)
                });

                let results = futures::future::join_all(futures).await;

                for result in results {
                    if let Some(desc) = result {
                        descriptions.push(desc);
                    }
                }

                if descriptions.is_empty() {
                    None
                } else {
                    Some(descriptions.join(" "))
                }
            }
        }

        impl DescriberDispatcher {
            pub fn all() -> Vec<Self> {
                vec![
                    $(
                        Self::$name($to),
                    )*
                ]
            }
        }
    };
}
