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
         * @brief 安全地读取输入图像像素值，处理边界情况（像素坐标越界）
         *
         * @param input 输入图像数据指针 (const float*)
         * @param x 要读取的像素的 x 坐标 (int)
         * @param y 要读取的像素的 y 坐标 (int)
         * @param width 输入图像宽度 (int)
         * @param height 输入图像高度 (int)
         * @return float 返回对应像素的值。如果坐标越界，则返回最近边界像素的值（边界像素拉伸/Clamp to edge）。
         */
        __device__ inline float readPixelSafe(const float* input, int x, int y, int width, int height) {
            // 1. 边界钳位(Clamping): 确保 x 和 y 坐标在有效范围内 [0, width-1] 和 [0, height-1]
            x = max(0, min(x, width - 1));
            y = max(0, min(y, height - 1));

            // 2. 计算一维索引并返回值
            return input[y * width + x];
        }

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
            // 1. 计算当前线程负责的输出像素坐标 (outX, outY)
            //    使用 gridDim, blockDim, blockIdx, threadIdx 计算全局线程索引
            const int outX = blockIdx.x * blockDim.x + threadIdx.x;
            const int outY = blockIdx.y * blockDim.y + threadIdx.y;

            // 2. 边界检查：确保当前线程处理的像素在输出图像范围内
            if (outX >= outWidth || outY >= outHeight) {
                return; // 超出范围的线程直接返回，不执行计算
            }

            // 3. 计算输出像素中心 (outX, outY) 对应到输入图像中的浮点坐标 (inX_f, inY_f)
            //    使用中心对齐的映射关系:
            //    in_coord = (out_coord + 0.5) * (in_dim / out_dim) - 0.5
            //    这样做可以确保输入和输出图像的中心点能够对齐。
            const float inX_f = (static_cast<float>(outX) + 0.5f) * (static_cast<float>(inWidth) / static_cast<float>(outWidth)) - 0.5f;
            const float inY_f = (static_cast<float>(outY) + 0.5f) * (static_cast<float>(inHeight) / static_cast<float>(outHeight)) - 0.5f;

            // 4. 确定用于插值的四个最近邻输入像素的整数坐标
            //    (x1, y1) 是左上角像素的坐标
            const int x1 = static_cast<int>(floorf(inX_f));
            const int y1 = static_cast<int>(floorf(inY_f));
            //    (x2, y2) 是右下角像素的坐标
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;

            // 5. 使用 readPixelSafe 安全地读取这四个像素的值
            //    readPixelSafe 会处理边界情况，无需在此处再次检查 x1, y1, x2, y2 是否越界
            const float p11 = readPixelSafe(input, x1, y1, inWidth, inHeight); // Top-left
            const float p21 = readPixelSafe(input, x2, y1, inWidth, inHeight); // Top-right
            const float p12 = readPixelSafe(input, x1, y2, inWidth, inHeight); // Bottom-left
            const float p22 = readPixelSafe(input, x2, y2, inWidth, inHeight); // Bottom-right

            // 6. 计算插值权重 (dx, dy)
            //    dx 是 inX_f 相对于 x1 的水平距离比例
            //    dy 是 inY_f 相对于 y1 的垂直距离比例
            const float dx = inX_f - static_cast<float>(x1);
            const float dy = inY_f - static_cast<float>(y1);

            // 7. 执行双线性插值计算
            //    首先在水平方向上进行线性插值
            const float interp1 = p11 * (1.0f - dx) + p21 * dx; // Top row interpolation
            const float interp2 = p12 * (1.0f - dx) + p22 * dx; // Bottom row interpolation
            //    然后在垂直方向上对水平插值的结果进行线性插值
            const float interpolatedValue = interp1 * (1.0f - dy) + interp2 * dy;

            // 8. 计算输出像素在一维数组中的索引
            const int outIndex = outY * outWidth + outX;

            // 9. 将计算得到的插值结果写入输出图像内存
            output[outIndex] = interpolatedValue;
        }
    "#,
}
