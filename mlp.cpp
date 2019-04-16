/*
Rede Neural Perceptron Multicamadas com 2 camdas processadoras
Clênio Eduardo 
*/
#include<bits/stdc++.h>

#define NEURONIOS_CAMADA_OCULTA 2
#define NEURONIOS_CAMADA_SAIDA 1
#define QTD_AMOSTRAS 4
#define QTD_DADOS 2
#define TAM_PESOS_OCULTA QTD_DADOS + 1 // mais um do bias
#define TAM_PESOS_SAIDA NEURONIOS_CAMADA_OCULTA + 1 //mais um do bias


using namespace std;

double u(double w[], double x[], double bias, int qtd_dados){

    double u = w[0] * bias;
    for(int i=0; i < qtd_dados; i++){
        u += x[i] * w[i + 1];
    }
    return u;
}

double sigmoide(double u){
    double e = 2.718281828459045235360287;
   

    return 1 / (1 + pow(e, -u));
}

double sigmoideDerivada(double y){   

    return y * (1 - y);

}

double deltaSaida(double d, double y){
    double erro = d - y;
    double derivada = sigmoideDerivada(y);

    return erro * derivada;
}

double deltaOculta(double y, double deltaP[],int indiceN, double wP[][TAM_PESOS_SAIDA], int qtdNeuronioP){

    double derivada = sigmoideDerivada(y);
    double sumDelta = 0;

    for(int j=0; j < qtdNeuronioP; j++){
        sumDelta += (deltaP[j] * wP[j][indiceN + 1]);
    }

    return derivada * sumDelta;

}

void inicializaPesos(double m[], int c){
    
    for(int i=0; i < c; i++){       
        m[i] = (double)(rand() % 100) /100;       
       
    }
}

int main(){

    srand((unsigned)time(NULL));

//Obter o conjunto de amostras de treinamento {x{k}}
    double x[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
//Associar o vetor de saida desejada {d{k}} para cada amostra
    double d[4][1] = {
        {0},
        {1},
        {1},
        {0}
    };

//Iniciar wji(1), wji(2) com valores aleatorios pequenos
double w1[NEURONIOS_CAMADA_OCULTA][TAM_PESOS_OCULTA]; //pesos da camada oculta
double w2[NEURONIOS_CAMADA_SAIDA][TAM_PESOS_SAIDA]; //pesos da camada de saída

double bias1[NEURONIOS_CAMADA_OCULTA];

//inicializando bias
for(int i=0; i < NEURONIOS_CAMADA_OCULTA; i++){
    bias1[i] = 1.0;
}

//inicializando bias
double bias2[NEURONIOS_CAMADA_SAIDA];

for(int i=0; i < NEURONIOS_CAMADA_SAIDA; i++){
    bias2[i] = 1.0;
}


double u1[NEURONIOS_CAMADA_OCULTA]; //potencial de ativação dos neuronios da camada oculta
double u2[NEURONIOS_CAMADA_SAIDA]; //potencial de ativação dos neuronios da camada de saida

double y1[NEURONIOS_CAMADA_OCULTA]; //saidas dos neuronio da camada oculta
double y2[NEURONIOS_CAMADA_SAIDA]; //saidas dos neuronios da camada de saída

double delta1[NEURONIOS_CAMADA_OCULTA]; //gradientes dos neuronios da camada oculta
double delta2[NEURONIOS_CAMADA_SAIDA]; //gradientes dos neuronios da camada de saída


//inicializando pesos da camada oculta
for(int l = 0; l < NEURONIOS_CAMADA_OCULTA; l++){
    inicializaPesos(w1[l], TAM_PESOS_OCULTA);   
}

//inicializando pesos da camada de saida
for(int l= 0; l < NEURONIOS_CAMADA_SAIDA; l++){
    inicializaPesos(w2[l], TAM_PESOS_SAIDA);
}


//Especificar a taxa de aprendizagem {n} e precisão requerida {e}
double n = 0.5;
double e = pow(10, -6);
long long epocas = 0;
double EQM;
double EQMatual = 0;
double EQManterior = 0;
double Eglobal, errinho;

do{
    EQManterior = EQMatual;
    EQM = 0;
    Eglobal = 0;

    //Inicio Fase Forward
    //Para todas as amostras de treinamento {x(k), d(k)}, fazer:
    for(int k = 0; k < QTD_AMOSTRAS; k++){
        errinho = 0;
        //potencial de ativação camda oculta
        for(int j=0; j < NEURONIOS_CAMADA_OCULTA; j++){
            u1[j] = u(w1[j], x[k], bias1[j], QTD_DADOS);
            y1[j] = sigmoide(u1[j]);
        }
        //potencial de ativação camada de saida
        for(int j=0; j < NEURONIOS_CAMADA_SAIDA; j++){
            u2[j] = u(w2[j], y1, bias2[j], NEURONIOS_CAMADA_OCULTA);
            y2[j] = sigmoide(u2[j]);

            errinho = errinho + pow(d[k][j] - y2[j], 2);
        }
        Eglobal = errinho / 2;
        EQM = EQM + Eglobal;
        //Fim da fase Forward

        //Inicio da fase Backward
        //determinar o gradiente da camda de saida
        for(int j=0; j < NEURONIOS_CAMADA_SAIDA; j++){
            delta2[j] =  deltaSaida(d[k][j], y2[j]);
        }
        
        //determinar o gradiente da camda oculta
        for(int j=0; j < NEURONIOS_CAMADA_OCULTA; j++){
            delta1[j] = deltaOculta(y1[j], delta2, j, w2, NEURONIOS_CAMADA_SAIDA); 
        }

        //Ajuste de pesos da camada de saida
        for(int j=0; j < NEURONIOS_CAMADA_SAIDA; j++){
            
            //atualizando o bias da camada de saída
            w2[j][0] += n * (delta2[j] * bias2[j]);
            //atualizando os pesos
            for(int l=1; l < TAM_PESOS_SAIDA; l++){
                w2[j][l] += n * (delta2[j] * y1[l-1]);
            }
        }

        //Ajuste de pesos da camada oculta
        for(int j=0; j < NEURONIOS_CAMADA_OCULTA; j++){

            //atualizando o bias da camada oculta
            w1[j][0] += n * (delta1[j] * bias1[j]);
            //atualizando os pesos
            for(int l=1; l < TAM_PESOS_OCULTA; l++){
                w1[j][l] += n * (delta1[j] * x[k][l-1]);
            }
        }
    }
    EQM = EQM / QTD_AMOSTRAS;
    EQMatual = EQM;
    epocas++;
    printf("EQM %lf \n", EQMatual);    

}while(abs(EQMatual - EQManterior) > e);



printf("REDE TREINADA COM %lld epocas \n", epocas); 

//-----------------------------------------------------------------------------------------------------------------------

//Fase de operação
//Para todas as amostras de treinamento {x(k), d(k)}, fazer:
    for(int k = 0; k < QTD_AMOSTRAS; k++){

        //potencial de ativação camda oculta
        for(int j=0; j < NEURONIOS_CAMADA_OCULTA; j++){
            u1[j] = u(w1[j], x[k], bias1[j], QTD_DADOS);
            y1[j] = sigmoide(u1[j]);
        }
        printf("\nAmostra %d ", k);
        //potencial de ativação camada de saida
        for(int j=0; j < NEURONIOS_CAMADA_SAIDA; j++){
            u2[j] = u(w2[j], y1, bias2[j], NEURONIOS_CAMADA_OCULTA);
            y2[j] = sigmoide(u2[j]);

            //printf("%lf ", y2[j]); 
            if(y2[j] > 0.5)
                printf("%lf ", 1.0);
            else
                printf("%lf ",0.0);           
        }
    }

    return 0;
}
