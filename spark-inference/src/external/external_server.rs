use std::io::Write;
use std::ops::Deref;
use zenoh::bytes::ZBytes;
use zenoh::Session;

pub struct ServerInput<'a> {
    pub image: &'a Vec<u8>,
    pub detect_result: Vec<DetectResult>,
}

pub struct DetectResult {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl<'a> ServerInput<'a> {
    pub fn new(image: &'a Vec<u8>, detect_result: Vec<DetectResult>) -> Self {
        Self {
            image,
            detect_result,
        }
    }

    fn to_zbytes(self) -> anyhow::Result<ZBytes> {
        let mut bytes = ZBytes::writer();
        static MAGIC: &[u8] = b"SPARK";
        bytes.write_all(MAGIC)?;

        bytes.write_all((self.image.len() as u32).to_le_bytes().as_slice())?;
        bytes.write_all(self.image)?;

        bytes.write_all((self.detect_result.len() as u32).to_le_bytes().as_slice())?;
        for result in &self.detect_result {
            bytes.write_all(&result.x.to_le_bytes())?;
            bytes.write_all(&result.y.to_le_bytes())?;
            bytes.write_all(&result.width.to_le_bytes())?;
            bytes.write_all(&result.height.to_le_bytes())?;
        }

        Ok(bytes.finish())
    }
}

pub struct ExternalServer {
    session: Session,
}

impl Deref for ExternalServer {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        &self.session
    }
}

type ZResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;

impl ExternalServer {
    pub async fn new() -> ZResult<Self> {
        let session = zenoh::open(zenoh::Config::default()).await?;
        Ok(Self { session })
    }
    pub async fn response(&self) -> ZResult<()> {
        let subscriber = self.declare_subscriber("spark/sam/response").await?;
        while let Ok(sample) = subscriber.recv_async().await {
            println!("Received: {:?}", sample);
        }

        Ok(())
    }
    pub async fn request(&self, server_input: ServerInput<'_>) -> ZResult<()> {
        self.put("spark/sam/request", uuid::Uuid::new_v4().into_bytes())
            .attachment(server_input.to_zbytes()?)
            .await?;

        Ok(())
    }
}
