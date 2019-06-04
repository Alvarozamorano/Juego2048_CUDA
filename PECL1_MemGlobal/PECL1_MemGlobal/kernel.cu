#include "../common/book.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <conio.h>
#include <Windows.h>
#include <time.h>


//Autores: Sanz Sacristán, Sergio y Zamorano Ortega, Álvaro
FILE *doc;
FILE *leer;

int columnas;
int filas;
char dificultad;
char modo;
int punt_record;

char pedirModo();
char pedirDificultad();
int pedirFilasTablero();
int pedirColumnasTablero();
int* generarMatriz();
void imprimirMatriz(int *matriz, int *numVidas, int *puntos);
bool imprimirEspacios(int x, int* matriz);
int contNum(int num);
int maxMatriz(int *matriz);
bool rellenarMatriz(int* matriz);
int recuento(int* matriz);
char comprobarPropiedades();
void jugar(int *matriz, int* numVidas, int* puntos);
void cargarPartida();
void guardarPartida(int* matriz, int* numVidas, int* puntos);
int sumaPuntAux(int tamaño, int* punt);
void guardarRecord();
void cargarRecord();


int main(void) {

	int *matriz;
	char cargarP;

	printf("Quieres cargar una partida anterior o empezar de nuevo ? (s: si | n : no)\n");
	fflush(stdin);
	scanf("%c", &cargarP);

	while (cargarP != 's' && cargarP != 'n') {
		printf("Introduce un valor valido para iniciar el juego\n");
		scanf("%c", &cargarP);
	}

	//Si no cargamos partida
	if (cargarP == 'n') {

		int vida = 5;
		int *numVidas;
		numVidas = &vida;

		int numPuntos = 0;
		int *puntos;
		puntos = &numPuntos;

		modo = pedirModo();//Pedimos modo
		dificultad = pedirDificultad();//Pedimos dificultad
		filas = pedirFilasTablero();//Pedimos filas
		columnas = pedirColumnasTablero();//Pedimos columnas

		printf("\nLos datos introducidos por el usuario son: %c %c %d %d\n", modo, dificultad, filas, columnas);
		char error = comprobarPropiedades();

		if (error == 'T') {//Si al comprobar propiedades nos da error
			goto Error;
		}

		matriz = generarMatriz();//Generamos la matriz del tablero
		rellenarMatriz(matriz);//La rellenamos de semillas
		cargarRecord();//Cargamos la puntuacion record

		getchar();
		getchar();
		system("cls");

		//Procedimiento para jugar al juego 
		jugar(matriz, numVidas, puntos);

		//Al terminar de jugar, guardamos record
		guardarRecord();

		printf("\n - - - - - - - - - - - - - - - - - - - - -");
		printf("\n - - - - - -  JUEGO TERMINADO  - - - - - - ");
		printf("\n - - - - - - - - - - - - - - - - - - - - - ");


	}
	else {
		//cargamos la partida
		cargarPartida();
		guardarRecord();
		printf("\n - - - - - - - - - - - - - - - - - - - - -");
		printf("\n - - - - - -  JUEGO TERMINADO  - - - - - - ");
		printf("\n - - - - - - - - - - - - - - - - - - - - - ");
	}
Error://Se sale si al comprobar propiedades nos da error
	getchar();
}

//Funcion que devuelve un error si las dimensiones de la matriz son demasiado grandes para la grafica
char comprobarPropiedades() {
	cudaDeviceProp prop;
	char error = 'F';
	int count;
	size_t globalMem;
	size_t hilosPorBloque;
	HANDLE_ERROR(cudaGetDeviceCount(&count));

	for (int i = 0; i < count; i++) {
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		globalMem = prop.totalGlobalMem;
		hilosPorBloque = prop.maxThreadsPerBlock;

		//Si el tamaño de la matriz supera las limitaciones de capacidad
		if ((filas*columnas * sizeof(int)) >= globalMem) {
			printf("La matriz solicitada ocupa %zd y excede la capacidad de memoria global de tu tarjeta gráfica que es %zd \n",
				filas*columnas * sizeof(int), globalMem);
			error = 'T';
		}
		//Si el numero de hilos supera las limitaciones de hilos por bloques
		if ((filas*columnas > hilosPorBloque)) {
			printf("La matriz solicitada ocupa %d hilos y excede la cantidad de hilos por bloque que es%zd \n",
				filas*columnas, hilosPorBloque);
			error = 'T';
		}

	}
	return error;

}


// GENERAR FUNCIONES BASICAS PARA EL TABLERO
//Generar matriz a 0
int *generarMatriz() {
	int* matriz = (int*)malloc(filas*columnas * sizeof(int));

	for (int i = 0; i < filas*columnas; i++) {
		matriz[i] = 0;
	}
	return matriz;
}

//Rellenar la matriz
bool rellenarMatriz(int* matriz) {
	bool terminado = false;
	int numSemillas;
	int numAleatorio;
	int random;
	time_t t;

	//Iniciamos el modo aleatorio
	srand((unsigned)time(&t));

	//Si la dificultad es facil
	if (dificultad == 'F') {
		numSemillas = 15; //introducimos 15 semillas
		if (recuento(matriz) < numSemillas) {
			terminado = true;
		}
		else {
			int posiblesNum[] = { 2, 4,8 }; //seleccionamos uno de estos números en cada semilla
			numAleatorio = 3;
			while (numSemillas > 0 && !terminado) {
				random = rand() % (filas*columnas);
				if (matriz[random] == 0) {
					matriz[random] = posiblesNum[rand() % numAleatorio]; //añadimos la nueva semilla en una posicion aleatoria
					numSemillas = numSemillas - 1;
				}
			}
			//Si hay menos espacios libres que 15 game over
			if (recuento(matriz) < 15) {
				terminado = true;
			}
		}
	}
	//Si la dificultad es dificil
	else {
		numSemillas = 8; //introducimos 8 semillas
		if (recuento(matriz) < numSemillas) {
			terminado = true;
		}
		else {
			int posiblesNum[] = { 2, 4 }; //seleccionamos uno de estos números en cada semilla
			numAleatorio = 2;
			while (numSemillas > 0 && !terminado) {
				random = rand() % (filas*columnas);
				if (matriz[random] == 0) {
					matriz[random] = posiblesNum[rand() % numAleatorio]; //añadimos la nueva semilla en una posicion aleatoria
					numSemillas = numSemillas - 1;
				}
			}
			//Si hay menos espacios libres que 8 game over
			if (recuento(matriz) < 8) {
				terminado = true;
			}
		}
	}
	return terminado;
}

//Cuenta el nº de 0s en la matriz
int recuento(int* matriz) {
	int recuento = 0;
	for (int i = 0; i < filas*columnas; i++) {
		if (matriz[i] == 0) {
			recuento = recuento + 1;
		}
	}
	return recuento;
}

//Metodo que solicita al usuario el modo
char pedirModo() {
	char modo = ' ';
	getchar();
	while (modo != 'M' && modo != 'A') {
		printf("Que modo desea para el juego? Automatico (A), Manual (M)\n");
		fflush(stdin);
		scanf("%c", &modo);
		if (modo != 'M' && modo != 'A') {
			printf("Usted ha introducido un modo que no existe: -%c.\n", modo);
			printf("Por favor, introduzca uno de los siguientes dmodos que se le presentan por pantalla.\n\n");
			scanf("%c", &modo);
		}
	}
	return modo;
}

//Metodo que solicita al usuario la dificultad
char pedirDificultad() {
	char dificultad = ' ';
	getchar();
	while (dificultad != 'F' && dificultad != 'D') {
		printf("Que dificultad desea para el juego? Facil (F), Dificil (D)\n");
		fflush(stdin);
		scanf("%c", &dificultad);
		if (dificultad != 'F' && dificultad != 'D') {
			printf("Usted ha introducido una dificutad que no existe: -%c.\n", dificultad);
			printf("Por favor, introduzca uno de las siguientes dificultades que se le presentan por pantalla.\n\n");
			scanf("%c", &dificultad);
		}
	}
	return dificultad;
}

//Metodo que solicita al usuario el numero de filas del tablero
int pedirFilasTablero() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int tamMaximo = prop.maxThreadsPerBlock;
	int filas;
	do {
		printf("\nIntroduzca las filas que tendra el tablero: ");
		fflush(stdin);
		scanf("%d", &filas);
		if (filas < 1 || filas > tamMaximo) {
			printf("Introduzca un numero de filas correcto\n");
		}
	} while (filas < 1); //El numero de filas tiene que ser un numero entero positivo

	return filas;
}

//Metodo para solicitar al usuario el numero de columnas del tablero
int pedirColumnasTablero() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int tamMaximo = prop.maxThreadsPerBlock;
	int columnas;

	do {
		printf("\nIntroduzca las columnas que tendra el tablero: ");
		fflush(stdin);
		scanf("%d", &columnas);
		if (columnas < 1 || columnas > tamMaximo) {
			printf("Introduzca un numero de columnas correcto\n");
		}
	} while (columnas < 1); //El numero de filas tiene que ser un numero entero positivo

	return columnas;
}

// FUNCIONES DE IMPRESIÓN DEL TABLERO
void imprimirMatriz(int *matriz, int *numVidas, int *puntos) {
	bool impar = false;
	printf("\nVIDAS: %d\n", *numVidas);
	printf("\nPUNTOS: %d", *puntos);
	printf("  RECORD: %d\n", punt_record);
	printf("\n\t|");
	for (int i = 0; i < filas*columnas; i++) {
		if ((i + 1) % columnas == 0) {
			if (matriz[i] == 0) {
				impar = imprimirEspacios(matriz[i], matriz);
				if (impar) {
					printf(" ");
				}
				printf(" ");
				imprimirEspacios(matriz[i], matriz);
				if (i == (filas*columnas) - 1) {
					printf("|\n\t");
				}
				else {
					printf("|\n\t|");
				}
			}
			else {
				impar = imprimirEspacios(matriz[i], matriz);
				if (impar) {
					printf(" ");
				}
				printf("%d", matriz[i]);
				imprimirEspacios(matriz[i], matriz);
				if (i == (filas*columnas) - 1) {
					printf("|\n\t");
				}
				else {
					printf("|\n\t|");
				}
			}
		}
		else {
			if (matriz[i] == 0) {
				impar = imprimirEspacios(matriz[i], matriz);
				if (impar) {
					printf(" ");
				}
				printf(" ");
				imprimirEspacios(matriz[i], matriz);
				printf("|");
			}
			else {
				impar = imprimirEspacios(matriz[i], matriz);
				if (impar) {
					printf(" ");
				}
				printf("%d", matriz[i]);
				imprimirEspacios(matriz[i], matriz);
				printf("|");
			}
		}
	}
	printf("\n");
}

bool imprimirEspacios(int x, int* matriz) {
	bool impar = false;
	int cifras = 0;
	int max = 0;
	int espacios;
	cifras = contNum(x);
	max = maxMatriz(matriz);
	espacios = contNum(max) - cifras;

	if (espacios % 2 != 0) {
		impar = true;
	}
	for (int i = 0; i < espacios / 2; i++) {
		printf(" ");
	}
	return impar;
}


int contNum(int num) {
	int contador = 0;
	while (num / 10 > 0) {
		num = num / 10;
		contador++;
	}
	return contador + 1;
}

int maxMatriz(int *matriz) {
	int max = 0;
	for (int i = 0; i < filas*columnas; i++) {
		if (max < matriz[i]) {
			max = matriz[i];
		}
	}
	return max;
}
// KERNELS
//Kernel que suma los elementos hacia la derecha
__global__ void sumarElementosDerecha(int *puntos, int *matriz, int numColumnas) {
	int pos = threadIdx.x;
	bool suma = false;
	bool terminado = false;

	int numElementos = 1;
	int posElementoSuma;

	//Las posiciones se recorren en la misma fila
	int i = 1;
	bool primero = true;

	do {
		//Si está en el limite, si su valor es 0, o si el valor de la posicion a estudiar es de dintinto valor
		if ((pos + 1) % numColumnas == 0 || matriz[pos] == 0 || (matriz[pos + i] != 0 && matriz[pos + i] != matriz[pos])) {
			terminado = true;
		}
		else {
			//Si son de igual valor
			if (matriz[pos] == matriz[pos + i]) {
				//Si es el primer elemento con igual valor
				if (primero) {
					posElementoSuma = pos + i;
					primero = false;
				}
				numElementos = numElementos + 1;
			}
			//Si es la columna limite
			if ((pos + 1 + i) % numColumnas == 0) {
				terminado = true;
			}
			i++;
		}
	} while (terminado == false);

	//Si el numero de iguales es par
	if (numElementos % 2 == 0) {
		suma = true;
	}

	__syncthreads();

	//Realiza la suma y cotabiliza los puntos
	if (suma) {
		matriz[posElementoSuma] = matriz[posElementoSuma] + matriz[pos];
		matriz[pos] = 0;
		puntos[pos] = matriz[posElementoSuma];
	}
	__syncthreads();
}

//Kernel que suma elementos hacia izquierda
__global__ void sumarElementosIzquierda(int *puntos, int *matriz, int numColumnas) {
	int pos = threadIdx.x;
	bool suma = false;
	bool terminado = false;

	int numElementos = 1;
	int posElementoSuma;

	//Las posiciones se recorren en la misma fila
	int i = 1;
	bool primero = true;

	do {
		//Si está en el limite, si su valor es 0, o si el valor de la posicion a estudiar es de dintinto valor
		if (pos % numColumnas == 0 || matriz[pos] == 0 || (matriz[pos - i] != 0 && matriz[pos - i] != matriz[pos])) {
			terminado = true;
		}
		else {
			//Si son de igual valor
			if (matriz[pos] == matriz[pos - i]) {
				//Si es el primer elemento con igual valor
				if (primero) {
					posElementoSuma = pos - i;
					primero = false;
				}
				numElementos = numElementos + 1;
			}
			//Si es la columna limite
			if ((pos - i) % numColumnas == 0) {
				terminado = true;
			}
			i++;
		}
	} while (terminado == false);

	//Si el numero de iguales es par
	if (numElementos % 2 == 0) {
		suma = true;
	}

	__syncthreads();

	//Realiza la suma y cotabiliza los puntos
	if (suma) {
		matriz[posElementoSuma] = matriz[posElementoSuma] + matriz[pos];
		matriz[pos] = 0;
		puntos[pos] = matriz[posElementoSuma];
	}
	__syncthreads();
}

//Kernel para sumar elementos hacia arriba
__global__ void sumarElementosArriba(int *puntos, int *matriz, int numColumnas) {
	int pos = threadIdx.x;
	bool suma = false;
	bool terminado = false;

	int numElementos = 1;
	int posElementoSuma;

	//Las posiciones se recorren mediante el numero de columnas
	int i = numColumnas;
	bool primero = true;

	do {
		//Si está en el limite, si su valor es 0, o si el valor de la posicion a estudiar es de dintinto valor
		if (pos < numColumnas || matriz[pos] == 0 || (matriz[pos - i] != 0 && matriz[pos - i] != matriz[pos])) {
			terminado = true;
		}
		else {
			//Si son de igual valor
			if (matriz[pos] == matriz[pos - i]) {
				//Si es el primer elemento con igual valor
				if (primero) {
					posElementoSuma = pos - i;
					primero = false;
				}
				numElementos = numElementos + 1;
			}
			//Si es la fila limite
			if ((pos - i) < numColumnas) {
				terminado = true;
			}
			i = i + numColumnas;
		}
	} while (terminado == false);

	//Si el numero de iguales es par
	if (numElementos % 2 == 0) {
		suma = true;
	}

	__syncthreads();

	//Realiza la suma y cotabiliza los puntos
	if (suma) {
		matriz[posElementoSuma] = matriz[posElementoSuma] + matriz[pos];
		matriz[pos] = 0;
		puntos[pos] = matriz[posElementoSuma];
	}
	__syncthreads();
}

//Kernel para sumar los elementos hacia abajo
__global__ void sumarElementosAbajo(int *puntos, int *matriz, int numFilas, int numColumnas) {
	int pos = threadIdx.x;
	bool suma = false;
	bool terminado = false;

	int numElementos = 1;
	int posElementoSuma;

	//Las posiciones se recorren mediante el numero de columnas
	int i = numColumnas;
	bool primero = true;

	do {
		//Si está en el limite, si su valor es 0, o si el valor de la posicion a estudiar es de dintinto valor
		if (pos >= (numColumnas*(numFilas - 1)) || matriz[pos] == 0 || (matriz[pos + i] != 0 && matriz[pos + i] != matriz[pos])) {
			terminado = true;
		}
		else {
			//Si son de igual valor
			if (matriz[pos] == matriz[pos + i]) {
				//Si es el primer elemento con igual valor
				if (primero) {
					posElementoSuma = pos + i;
					primero = false;
				}
				numElementos = numElementos + 1;
			}
			//Si es la fila limite
			if ((pos + i) >= (numColumnas*(numFilas - 1))) {
				terminado = true;
			}
			i = i + numColumnas;
		}
	} while (terminado == false);

	//Si el numero de iguales es par
	if (numElementos % 2 == 0) {
		suma = true;
	}

	__syncthreads();

	//Realiza la suma y cotabiliza los puntos
	if (suma) {
		matriz[posElementoSuma] = matriz[posElementoSuma] + matriz[pos];
		matriz[pos] = 0;
		puntos[pos] = matriz[posElementoSuma];
	}
	__syncthreads();
}

//Kernel que mueve elementos hacia la derecha
__global__ void moverElementosDerecha(int *matriz, int numFilas, int numColumnas) {
	int pos = threadIdx.x;
	bool mov = false;
	bool terminado = false;

	int numElementos = 0;
	int i = 1; //Las posiciones se recorren en la misma fila
	int valor = matriz[pos];

	do {
		//Si el hilo está la columna limite
		if ((pos + 1) % numColumnas == 0) {
			terminado = true;
		}
		else {
			//Si el valor del hilo es 0
			if (matriz[pos] == 0) {
				terminado = true;
			}
			else {
				//Si se encuentra un 0
				if (matriz[pos + i] == 0) {
					numElementos = numElementos + 1;
				}

				//Si la posicion a estudiar es el limite
				if ((pos + i + 1) % numColumnas == 0) {
					if (numElementos > 0) {
						mov = true;
					}
					terminado = true;
				}
			}
			i++;
		}
	} while (terminado == false);

	__syncthreads();

	//Pone a 0 su posición
	if (mov) {
		matriz[pos] = 0;

	}
	__syncthreads();

	//Mueve el valor a la posición correspondiente
	if (mov) {
		matriz[pos + numElementos] = valor;
	}

}

//Kernel que mueve elementos hacia la izquierda
__global__ void moverElementosIzquierda(int *matriz, int numFilas, int numColumnas) {
	int pos = threadIdx.x;
	bool mov = false;
	bool terminado = false;
	int valor = matriz[pos];

	int numElementos = 0;
	int i = 1; //Las posiciones se recorren en la misma fila

	do {
		//Si el hilo está la columna limite
		if (pos % numColumnas == 0) {
			terminado = true;
		}
		else {
			//Si el valor del hilo es 0
			if (matriz[pos] == 0) {
				terminado = true;
			}
			else {
				//Si se encuentra un 0
				if (matriz[pos - i] == 0) {
					numElementos = numElementos + 1;
				}

				//Si la posicion a estudiar es el limite
				if ((pos - i) % numColumnas == 0) {
					if (numElementos > 0) {
						mov = true;
					}
					terminado = true;
				}
			}
			i++;
		}
	} while (terminado == false);

	__syncthreads();

	//Pone a 0 su posición
	if (mov) {
		matriz[pos] = 0;
	}
	__syncthreads();

	//Mueve el valor a la posición correspondiente
	if (mov) {
		matriz[pos - numElementos] = valor;
	}
}

//Kernel que mueve elementos hacia arriba
__global__ void moverElementosArriba(int *matriz, int numFilas, int numColumnas) {
	int pos = threadIdx.x;
	bool mov = false;
	bool terminado = false;
	int valor = matriz[pos];
	int numElementos = 0;
	int i = numColumnas; //Las posiciones se recorren mediante el numero de columnas

	do {
		//Si el hilo se encuentra en la fila limite
		if (pos < numColumnas) {
			terminado = true;
		}
		else {
			//Si el valor del hilo es 0
			if (matriz[pos] == 0) {
				terminado = true;
			}
			else {
				//Si se encuentra un 0
				if (matriz[pos - i] == 0) {
					numElementos = numElementos + 1;
				}

				//Si la posicion a estudiar es la fila limite
				if ((pos - i) < numColumnas) {
					if (numElementos > 0) {
						mov = true;
					}
					terminado = true;
				}
			}
			i = i + numColumnas;
		}
	} while (terminado == false);

	__syncthreads();

	//Pone a 0 su posición
	if (mov) {
		matriz[pos] = 0;
	}
	__syncthreads();

	//Mueve el valor a la posición correspondiente
	if (mov) {
		matriz[pos - (numElementos * numColumnas)] = valor;
	}
}

//Kernel que mueve elemenetos hacia abajo
__global__ void moverElementosAbajo(int *matriz, int numFilas, int numColumnas) {
	int pos = threadIdx.x;
	bool mov = false;
	bool terminado = false;
	int valor = matriz[pos];
	int numElementos = 0;
	int i = numColumnas; //Las posiciones se recorren mediante el numero de columnas

	do {
		//Si el hilo se encuentra en la fila limite
		if (pos >= numColumnas * (numFilas - 1)) {
			terminado = true;
		}
		else {
			//Si el valor del hilo es 0
			if (matriz[pos] == 0) {
				terminado = true;
			}
			else {
				//Si se encuentra un 0
				if (matriz[pos + i] == 0) {
					numElementos = numElementos + 1;
				}

				//Si la posicion a estudiar es la fila limite
				if ((pos + i) >= numColumnas * (numFilas - 1)) {
					if (numElementos > 0) {
						mov = true;
					}
					terminado = true;
				}
			}
			i = i + numColumnas;
		}
	} while (terminado == false);

	__syncthreads();

	//Pone a 0 su posición
	if (mov) {
		matriz[pos] = 0;
	}

	__syncthreads();

	//Mueve el valor a la posición correspondiente
	if (mov) {
		matriz[pos + (numElementos * numColumnas)] = valor;
	}
}

//Metodo que suma los puntos de cada celda sumada
int sumaPuntAux(int tamaño, int* punt) {
	int suma = 0;
	for (int i = 0; i < tamaño; i++) {
		suma += punt[i];
	}
	return suma;
}

//Funcion que simula el juego
void jugar(int *matriz, int* numVidas, int* puntos) {
	int *dev_matriz;
	int *dev_puntos;
	char movimiento = ' ';
	int numRan;
	bool terminado;
	int* puntos_aux = generarMatriz();
	time_t t;

	srand((unsigned)time(&t));

	while (*numVidas > 0) {
		terminado = false;
		imprimirMatriz(matriz, numVidas, puntos);
		while (!terminado) {
			if (modo == 'M') {
				printf("Pulse una flecha...Pulse g para guardar");
				bool bucle = true;

				while (bucle) {
					movimiento = _getch();
					switch (movimiento) {
					case 72:
						movimiento = 'W'; //Arriba
						bucle = false;
						break;
					case 80:
						movimiento = 'S'; //Abajo
						bucle = false;
						break;
					case 75:
						movimiento = 'A'; //Izquierda
						bucle = false;
						break;
					case 77:
						movimiento = 'D'; //Derecha
						bucle = false;
						break;
					case 103:
						//SI PULSAS G GUARDA LA PARTIDA
						guardarPartida(matriz, numVidas, puntos);
						break;
					}
				}
			}
			else {
				numRan = rand() % 4;
				switch (numRan) {
				case 0:
					movimiento = 'W'; //Arriba
					break;
				case 1:
					movimiento = 'A'; //Izquierda
					break;
				case 2:
					movimiento = 'S'; //Abajo
					break;
				case 3:
					movimiento = 'D'; //Derecha
					break;
				}
			}

			//Inicializamos a 0 los elementos del vector de puntos
			for (int i = 0; i < filas*columnas; i++) {
				puntos_aux[i] = 0;
			}

			//Reservamos posiciones de memoria y copiamos de host a device

			cudaMalloc((void**)&dev_matriz, filas*columnas * sizeof(int));

			cudaMemcpy(dev_matriz, matriz, filas*columnas * sizeof(int), cudaMemcpyHostToDevice);

			cudaMalloc((void**)&dev_puntos, filas*columnas * sizeof(int));

			cudaMemcpy(dev_puntos, puntos_aux, filas*columnas * sizeof(int), cudaMemcpyHostToDevice);

			//Llamamos a los kernels correspondientes dependiendo del movimiento
			switch (movimiento) {
			case 'W':
				printf("\n\nARRIBA");
				sumarElementosArriba << < 1, filas*columnas >> > (dev_puntos, dev_matriz, columnas);
				moverElementosArriba << < 1, filas*columnas >> > (dev_matriz, filas, columnas);
				break;
			case 'A':
				printf("\n\nIZQUIERDA");
				sumarElementosIzquierda << < 1, filas*columnas >> > (dev_puntos, dev_matriz, columnas);
				moverElementosIzquierda << < 1, filas*columnas >> > (dev_matriz, filas, columnas);
				break;
			case 'S':
				printf("\n\nABAJO");
				sumarElementosAbajo << < 1, filas*columnas >> > (dev_puntos, dev_matriz, filas, columnas);
				moverElementosAbajo << < 1, filas*columnas >> > (dev_matriz, filas, columnas);
				break;
			case 'D':
				printf("\n\nDERECHA");
				sumarElementosDerecha << < 1, filas*columnas >> > (dev_puntos, dev_matriz, columnas);
				moverElementosDerecha << < 1, filas*columnas >> > (dev_matriz, filas, columnas);
				break;
			}

			//Recuperamos los datos del device
			cudaMemcpy(matriz, dev_matriz, filas*columnas * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(puntos_aux, dev_puntos, filas*columnas * sizeof(int), cudaMemcpyDeviceToHost);

			//Se suman los puntos
			*puntos += sumaPuntAux(filas*columnas, puntos_aux);

			//Si los puntos son mayores que el record
			if (*puntos > punt_record) {
				punt_record = *puntos;
			}

			terminado = rellenarMatriz(matriz);
			imprimirMatriz(matriz, numVidas, puntos);

			cudaFree(dev_matriz);

			//Si no se pueden meter tantas semillas que establece el modo seleccionado
			if (terminado) {
				*numVidas = *numVidas - 1;
				printf("\t\tGAME OVER. Pulsa ENTER");
				getchar();
			}
		}
		//Liberamos memoria
		free(matriz);

		//Si todavia quedan vidas
		if (*numVidas > 0) {
			matriz = generarMatriz();//Se genera otro tablero
			rellenarMatriz(matriz);
		}
		else {
			printf("\nJUEGO TERMINADO. SIN VIDAS");
		}
		system("cls");
	}
	system("cls");
}

//Funcion para cargar una partida guardada en el txt guardado
void cargarPartida() {

	leer = fopen("guardado.txt", "r");

	//Punteros;
	int vida;
	int *numVidas = NULL;
	numVidas = &vida;

	int puntos;
	int *numPuntos = NULL;
	numPuntos = &puntos;

	//leer variables del txt
	fscanf(leer, "%d", &filas);
	printf("\nFILAS: %d", filas);

	fscanf(leer, "%d", &columnas);
	printf("\nCOLUMNAS: %d", columnas);

	fscanf(leer, "%hhd", &dificultad);
	printf("\nDIFICULTAD: %c", dificultad);

	fscanf(leer, "%hhd", &modo);
	printf("\nMODO: %c", modo);

	fscanf(leer, "%d", &vida);
	printf("\nNUMERO DE VIDAS: %d", vida);

	fscanf(leer, "%d", &puntos);
	printf("\nPUNTOS: %d", puntos);

	int* matriz = (int*)malloc(filas*columnas * sizeof(int));

	for (int i = 0; i < filas*columnas; i++) {
		fscanf(leer, "%d", &matriz[i]);
	}

	cargarRecord();//Cargamos record
	jugar(matriz, numVidas, numPuntos);
}

//Metodo para guardar la partida en txt
void guardarPartida(int* matriz, int* numVidas, int* puntos) {

	doc = fopen("guardado.txt", "w");
	fprintf(doc, "%i \n", filas);
	fprintf(doc, "%i \n", columnas);
	fprintf(doc, "%i \n", dificultad);
	fprintf(doc, "%i \n", modo);
	fprintf(doc, "%i \n", *numVidas);
	fprintf(doc, "%i \n", *puntos);
	for (int i = 0; i < (filas*columnas); i++) {
		fprintf(doc, "%i ", matriz[i]);
	}
	fclose(doc);

	printf("\n--GUARDADO--\n");
}

//Funcion para cargar el record
void cargarRecord() {

	leer = fopen("record.txt", "r");

	fscanf(leer, "%d", &punt_record);
}

//Funcion para guardar el record
void guardarRecord() {
	doc = fopen("record.txt", "w");
	fprintf(doc, "%i \n", punt_record);
	fclose(doc);
	printf("\n--GUARDADO RECORD--\n");
}