new_cuda! {
    normalise_pixel_mean => r#"
        extern "C" __global__ void normalise_pixel_mean(
            float *out, const unsigned char *inp,
            const float *mean, const float *std,
            const size_t numel
        ) {
            // 计算当前线程的索引
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            // 每个线程处理一个像素的三个通道（RGB）
            unsigned int idx = i * 3; // 计算该像素的RGB起始位置
            // 分别处理 R, G, B 通道并进行归一化处理
            out[i] =                 ((static_cast<float>(inp[idx]) / 255.0f) - mean[0]) / std[0];       // 处理 R 通道
            out[i + numel / 3] =     ((static_cast<float>(inp[idx + 1]) / 255.0f) - mean[1]) / std[1];  // 处理 G 通道
            out[i + 2 * numel / 3] = ((static_cast<float>(inp[idx + 2]) / 255.0f) - mean[2]) / std[2]; // 处理 B 通道
        }
    "#,
    normalise_pixel_div => r#"
        extern "C" __global__ void normalise_pixel_div(float *out, const unsigned char *inp, const size_t numel) {
            // 计算当前线程的索引
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            // 每个线程处理一个像素的三个通道（RGB）
            unsigned int idx = i * 3; // 计算该像素的RGB起始位置
            // 分别处理 R, G, B 通道并进行归一化处理
            out[i] = static_cast<float>(inp[idx]) / 255.0f;       // 处理 R 通道
            out[i + numel / 3] = static_cast<float>(inp[idx + 1]) / 255.0f;  // 处理 G 通道
            out[i + 2 * numel / 3] = static_cast<float>(inp[idx + 2]) / 255.0f; // 处理 B 通道
        }
    "#,
}
