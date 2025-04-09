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
    bilinear_interpolate_centered => r#"
        /**
         * @brief 使用双线性插值调整图像大小的 CUDA 核函数 (中心对齐)
         *
         * @param input 指向输入图像数据的常量指针 (设备内存)
         * @param output 指向输出图像数据的指针 (设备内存)
         * @param inWidth 输入图像的宽度
         * @param inHeight 输入图像的高度
         * @param outWidth 输出图像的宽度
         * @param outHeight 输出图像的高度
         */
        extern "C" __global__ void bilinear_interpolate_centered(
            const float* input, float* output,
            int inWidth, int inHeight,
            int outWidth, int outHeight
        ) {
            // Calculate the global thread indices for the output pixel (j corresponds to width, i corresponds to height)
            int j = blockIdx.x * blockDim.x + threadIdx.x; // Output column index
            int i = blockIdx.y * blockDim.y + threadIdx.y; // Output row index

            // Check if the thread is within the bounds of the output image
            if (j < outWidth && i < outHeight) {
                // --- Map output coordinates (i, j) to input coordinates (y, x) ---
                // Note: This mapping follows the logic of your Rust code.
                // It maps the discrete output grid points [0, outWidth-1] to the continuous input range [0, inWidth-1].
                float x = (float)j / (float)outWidth * (float)(inWidth - 1);
                float y = (float)i / (float)outHeight * (float)(inHeight - 1);

                // --- Find the 4 neighboring pixels in the input image ---
                int x0 = floorf(x);
                int x1 = ceilf(x);
                int y0 = floorf(y);
                int y1 = ceilf(y);

                // --- Clamp coordinates to input image boundaries ---
                // This prevents reading outside the input buffer, which is crucial!
                x0 = max(0, min(x0, inWidth - 1));
                x1 = max(0, min(x1, inWidth - 1));
                y0 = max(0, min(y0, inHeight - 1));
                y1 = max(0, min(y1, inHeight - 1));

                // Alternative using fmaxf/fminf:
                // x0 = fmaxf(0.0f, fminf((float)x0, (float)(inWidth - 1)));
                // x1 = fmaxf(0.0f, fminf((float)x1, (float)(inWidth - 1)));
                // y0 = fmaxf(0.0f, fminf((float)y0, (float)(inHeight - 1)));
                // y1 = fmaxf(0.0f, fminf((float)y1, (float)(inHeight - 1)));

                // --- Get the values of the 4 neighboring pixels ---
                // Remember: Access pattern is input[row * width + col]
                float p00 = input[y0 * inWidth + x0];
                float p01 = input[y0 * inWidth + x1];
                float p10 = input[y1 * inWidth + x0];
                float p11 = input[y1 * inWidth + x1];

                // --- Calculate interpolation weights ---
                float dx = x - (float)x0;
                float dy = y - (float)y0;

                // --- Perform bilinear interpolation ---
                float interpolated_value =
                    p00 * (1.0f - dx) * (1.0f - dy) +
                    p01 * dx * (1.0f - dy) +
                    p10 * (1.0f - dx) * dy +
                    p11 * dx * dy;

                // --- Write the result to the output image ---
                // Access pattern: output[row * width + col]
                output[i * outWidth + j] = interpolated_value;
            }
        }
    "#,
}
