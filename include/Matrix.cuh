#pragma once

template<typename T>
class Matrix
{
public:
    __device__ __host__ Matrix() = default;
    __device__ __host__ Matrix(const Matrix &) = default;
    __device__ __host__ Matrix& operator=(const Matrix &) = default;
    __device__ __host__ Matrix(T *_data,int _rows,int _cols,int _strideOfRow,int _strideOfCol):
            data(_data),
            rows(_rows),
            cols(_cols),
            strideOfRow(_strideOfRow),
            strideOfCol(_strideOfCol){}

    // 返回该矩阵所有字节数
    constexpr __device__ __host__ int GetNumberOfBytes() const
    {
        return rows*cols*sizeof(T);
    }

    // 返回该矩阵元素个数
    constexpr __device__ __host__ int GetNumberOfElements() const
    {
        return rows*cols;
    }

    // 访问某个元素，该元素的索引为二维逻辑索引：(rowIndex,colIndex)
    __device__ __host__ float &operator()(int rowIndex,int colIndex)
    {
        // 计算内存索引
        int memoryIndex=rowIndex*strideOfRow+colIndex*strideOfCol;

        return data[memoryIndex];
    }

    // 访问某个元素，该元素的索引为一维逻辑索引：(Index)
    __device__ __host__ float &operator()(int index)
    {
        // 转换为二维逻辑索引
        int colIndex=index%cols;
        int rowIndex=index/cols;

        // 计算内存索引
        int memoryIndex=rowIndex*strideOfRow+colIndex*strideOfCol;

        return data[memoryIndex];
    }



public:
    T *data = nullptr; // 数据指针
    int rows = 0;// 矩阵的行数
    int cols = 0;// 矩阵的列数
    int strideOfRow = 0;// 行步长
    int strideOfCol = 0;// 列步长

};
