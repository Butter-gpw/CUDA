# SGEMM
参考资料：
- [通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)
- [矩阵乘法的 CUDA 实现、优化及性能分析](https://chiemon.github.io/2020/02/06/CUDA-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95-%E4%BC%98%E5%8C%96%E5%8F%8A%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90-%E4%B8%8A.html)

通用矩阵乘法 (General Matrix Multiplication，GEMM)，的定义为： $ C \leftarrow \alpha AB + \beta C$
## CPU