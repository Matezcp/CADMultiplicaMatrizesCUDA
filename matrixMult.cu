#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void multiplicaMatriz(double *matrizACuda, double *matrizBCuda, double *matrizCCuda, int n) 
{
    __shared__ double subMatrizA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double subMatrizB[BLOCK_SIZE][BLOCK_SIZE];

    int linha = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int coluna = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idx;
    double calculo = 0;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = linha * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            subMatrizA[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            subMatrizA[threadIdx.y][threadIdx.x] = matrizACuda[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + coluna;
        if(idx >= n*n)
        {
            subMatrizB[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            subMatrizB[threadIdx.y][threadIdx.x] = matrizBCuda[idx];
        }
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            calculo += subMatrizA[threadIdx.y][k] * subMatrizB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(linha < n && coluna < n)
    {
        matrizCCuda[linha * n + coluna] = calculo;
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
    
    //Calcula a carga de trabalho
    unsigned int carga_trabalho = (tam + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //Define nossas threads e nossos blocos
    dim3 dimGrid(carga_trabalho, carga_trabalho);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    //Chama a função para fazer a multiplicação
    multiplicaMatriz<<<dimGrid, dimBlock>>>(matrizACuda, matrizBCuda, matrizCCuda, tam);    

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
