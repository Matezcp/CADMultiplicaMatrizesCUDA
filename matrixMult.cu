
#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void gpu_square_matrix_mult(double *d_a, double *d_b, double *d_result, int n) 
{
    __shared__ double tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}


int main(int argc,char **argv){
    //Declara as matrizes que irão para a GPU
    double *matrizACuda,*matrizBCuda,*matrizCCuda;
    //Declara as matrizes que ficarão na CPU
    double *matrizA,*matrizB,*matrizC; 
    //Declara as variáveis de tamanho e índice
    int tam,i,j,k; 

    //Lê a dimensão da matriz
    fscanf(stdin,"%d\n",&tam); 

    //Aloca as matrizes
    matrizA=(double*)malloc(tam*tam*sizeof(double));
    matrizB=(double*)malloc(tam*tam*sizeof(double));
    matrizC=(double*)malloc(tam*tam*sizeof(double));
    cudaMalloc((void **) &matrizACuda, sizeof(double)*tam*tam);
    cudaMalloc((void **) &matrizBCuda, sizeof(double)*tam*tam);
    cudaMalloc((void **) &matrizCCuda, sizeof(double)*tam*tam);

    //Lê as matrizes A e B
    for(i=0;i<tam;i++)
        for(j=0;j<tam;j++)
            fscanf(stdin, "%lf ", &matrizA[i * tam + j]);
    for(i=0;i<tam;i++)
        for(j=0;j<tam;j++)
            fscanf(stdin,"%lf ",&matrizB[i*tam+j]);
    
    //Envia do host para o Device
    cudaMemcpy(matrizACuda, matrizA, tam*tam*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(matrizBCuda, matrizB, tam*tam*sizeof(double),cudaMemcpyHostToDevice);

    //Calcula C=A*B
    /*for(i=0;i<tam;i++)
        for(j=0;j<tam;j++)
            for(k=0;k<tam;k++)
                matrizC[i*tam+j]+=matrizA[i*tam+k]*matrizB[k*tam+j];*/
    
    unsigned int grid_rows = (tam + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (tam + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(matrizACuda, matrizBCuda, matrizCCuda, tam);    

    //Envia a resposta do Device para o Host
    cudaMemcpy(matrizC, matrizCCuda, sizeof(double)*tam*tam, cudaMemcpyDeviceToHost);

    //Imprime o resultado    
    for(i=0;i<tam;i++){
        for(j=0;j<tam;j++)
            printf("%.1lf ",matrizC[i*tam+j]);
        printf("\n");
    }
    
    //Desaloca as matrizes
    cudaFree(matrizACuda);
    cudaFree(matrizBCuda);
    cudaFree(matrizCCuda);
    cudaFreeHost(matrizA);
    cudaFreeHost(matrizB);
    cudaFreeHost(matrizC);

    return 0;
}
