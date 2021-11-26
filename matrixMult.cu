#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>

#define THREADSPERBLOCK 8

__global__ void multiplicaMatriz(double *matrizACuda, double *matrizBCuda, double *matrizCCuda, int tam) 
{
    /*Utilizamos 2 sub matrizes, que irão armazenar os valores das matrizes A e B necessários
    para nosso cálculo, essas variáveis são compartilhadas entre as threads*/
    __shared__ double subMatrizA[THREADSPERBLOCK][THREADSPERBLOCK];
    __shared__ double subMatrizB[THREADSPERBLOCK][THREADSPERBLOCK];

    //Calculamos qual linha é de nossa responsabilidade
    int linha = blockIdx.x * THREADSPERBLOCK + threadIdx.x;
    //Calculamos qual coluna é de nossa responsabilidade
    int coluna = blockIdx.y * THREADSPERBLOCK + threadIdx.y;
    
    //Variavel que armazenará o valor calculado
    double calculo = 0;


    //Faz o calculo o número de vezes necessário (basicamente iterando entre as grades)
    for (int pulo = 0; pulo < gridDim.x; ++pulo) 
    {
        //Calcula a posição do valor que iremos pegar da matriz A
        int idx = linha * tam + pulo * THREADSPERBLOCK + threadIdx.y;
        //Se a posição ultrapassar o limite, apenas colocamos 0 em nossa sub matriz
        if(idx >= tam*tam)
            subMatrizA[threadIdx.x][threadIdx.y] = 0;
        //Caso contrário colocamos o valor em nossa sub matriz
        else
            subMatrizA[threadIdx.x][threadIdx.y] = matrizACuda[idx];
        //Calcula a posição do valor que iremos pegar da matriz B
        idx = (pulo * THREADSPERBLOCK + threadIdx.x) * tam + coluna;
        //Se a posição ultrapassar o limite, apenas colocamos 0 em nossa sub matriz
        if(idx >= tam*tam)
            subMatrizB[threadIdx.x][threadIdx.y] = 0;
        //Caso contrário colocamos o valor em nossa sub matriz
        else
            subMatrizB[threadIdx.x][threadIdx.y] = matrizBCuda[idx];

        //É necessário haver uma sincronização das threads para somarmos a resposta, por conta de nossas variáveis compartilhadas
        __syncthreads();
        //É feito o calculo do valor
        for (int k = 0; k < THREADSPERBLOCK; ++k) 
            calculo += subMatrizA[threadIdx.x][k] * subMatrizB[k][threadIdx.y];

        //Aguarda as threads sincronizarem novamente antes de começar uma nova iteração
        __syncthreads();
    }
    //Se estiver tudo correto com nossos indices de linha e coluna atualizamos o valor da matriz resultado C
    if(linha < tam && coluna < tam)
        matrizCCuda[linha * tam + coluna] = calculo;
}


int main(int argc,char **argv){
    //Declara as matrizes que irão para a GPU
    double *matrizACuda,*matrizBCuda,*matrizCCuda;
    //Declara as matrizes que ficarão na CPU
    double *matrizA,*matrizB,*matrizC; 
    //Declara as variáveis de tamanho e índice
    int tam,i,j;

    //Lê a dimensão da matriz
    fscanf(stdin,"%d\n",&tam); 

    //Aloca as matrizes do host
    cudaMallocHost((void**)&matrizA,tam*tam*sizeof(double));
    cudaMallocHost((void**)&matrizB,tam*tam*sizeof(double));
    cudaMallocHost((void**)&matrizC,tam*tam*sizeof(double));
    //Aloca as matrizes do device
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
    
    //Calcula a quantidade de blocos por grade (haverão tam threads por grade)
    int blocksPerGrid = (tam+THREADSPERBLOCK-1)/THREADSPERBLOCK;
    //Define nossas threads e nossos blocos
    dim3 dimGrid(blocksPerGrid,blocksPerGrid);
    dim3 dimBlock(THREADSPERBLOCK,THREADSPERBLOCK);

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
    
    //Desaloca as matrizes do device
    cudaFree(matrizACuda);
    cudaFree(matrizBCuda);
    cudaFree(matrizCCuda);
    //Desaloca as matrizes do host
    cudaFreeHost(matrizA);
    cudaFreeHost(matrizB);
    cudaFreeHost(matrizC);

    return 0;
}
