#include <fstream>
#include <iostream>
#include <math.h>
#include <map>
#include <vector>
#include <set>
#include <string>
#include <iomanip>	// Fijar largo de decimales.

/*
	Implementación de cálculo de correlación espacio temporal propuesto en:
	Measuring spatio-temporal autocorrelation in time series data of collective human mobility 
	[Gao, Cheng, Meng and Liu - Geo-spatial information science, 2019]
	Aplicado a datos de sensores de tráfico de madrid
	Al cargar la lista se considera la vecindad como SIMÉTRICA.
	Proceso de carga de datos:
		- Primero se cargan las series de tiempo.
		- Luego se identifican y marcan las series fijas.
		- Se cargan las listas de vecindad.
		- Se crea la lista de vecindad simétrica descartando aquellos sensores son series fijas.
		- Se calcula el índice de autocorrelación espacio temporal.
*/

using namespace std;

string print_celda(int x, int y, float v){
	string s = "";
	s += "[" + to_string(x) + "][" + to_string(y) + "] (" + to_string(v) + ")";
	return s;
}

string print_serie(vector<float> v){
	string s = "";
	for(int i=0; i<v.size(); i++){
		s += to_string(v[i]) + " ";
	}
	return s;
}

string print_serie(vector<int> v){
	string s = "";
	for(int i=0; i<v.size(); i++){
		s += to_string(v[i]) + " ";
	}
	return s;
}

float promedioST(vector<int> x){
	unsigned long long acum = 0;
	for(int i=0; i<x.size(); i++){
		acum += x[i];
	}
	acum /= x.size();
	return acum;
}

double ZETA(vector<float> X, vector<float> Y){
	/*
		- First CORT is calculated: 
			Adaptative temporal dissimilarity:
			covers temporal correlation and data distance
			Result in range -1 to 1:
				-1 X and Y share similar growth in rate but opposite in direction
				1 X and Y have a similiar growth in rate and direction
				0 different temporal behaviors
		- Then PHI is applied:
			tunes the first order correlation coefficient from [-1,1] to [0,2] where
			phi(0) = 1, phi(1) = 0, and phi(-1) = 2.
			Then, less value means more similar time series, 
			and greater values means less similar time series.
		- Finally is multiplied by the difference of accumulative volumes
	*/
	double numeradorCORT = 0.0;
	double denCORTFactorX = 0.0;
	double denCORTFactorY = 0.0;
	double auxDiffX, auxDiffY;
	double accVolX = 0.0 + X[0];
	double accVolY = 0.0 + X[0];
	for(int i=1; i <X.size(); i++){
		auxDiffX = X[i]-X[i-1];
		auxDiffY = Y[i]-Y[i-1];
		numeradorCORT += (auxDiffX*auxDiffY);
		denCORTFactorX += (auxDiffX*auxDiffX);
		denCORTFactorY += (auxDiffY*auxDiffY);
		accVolX += X[i];
		accVolY += Y[i];
	}
	double denominadorCORT = sqrt(denCORTFactorX) * sqrt(denCORTFactorY);
	double CORT = numeradorCORT / denominadorCORT;
	double PHI = 2 / (1 + exp(2*CORT));
	double ZETA = PHI * (accVolX - accVolY);
//    if(denCORTFactorX * denCORTFactorY == 0){
//        cout << "Denominador en 0: " << auxDiffX << " " << auxDiffY << endl;
//        cout << print_serie(X) << endl;
//        cout << print_serie(Y) << endl;
//    }
	return ZETA;
}

double ZETA(vector<int> X, vector<float> Y){
	/*
		Igual a la función anterior pero para valores enteros
	*/
	double numeradorCORT = 0.0;
	double denCORTFactorX = 0.0;
	double denCORTFactorY = 0.0;
	double auxDiffX, auxDiffY;
	double accVolX = 0.0 + X[0];
	double accVolY = 0.0 + X[0];
	for(int i=1; i <X.size(); i++){
		auxDiffX = X[i]-X[i-1];
		auxDiffY = Y[i]-Y[i-1];
		numeradorCORT += (auxDiffX*auxDiffY);
		denCORTFactorX += (auxDiffX*auxDiffX);
		denCORTFactorY += (auxDiffY*auxDiffY);
		accVolX += X[i];
		accVolY += Y[i];
	}
	double denominadorCORT = sqrt(denCORTFactorX) * sqrt(denCORTFactorY);
	double CORT = numeradorCORT / denominadorCORT;
	double PHI = 2 / (1 + exp(2*CORT));
	double ZETA = PHI * (accVolX - accVolY);
	return ZETA;
}

template <typename TipoNumero>
void calculaIAET(const vector<vector<TipoNumero>>& datos, vector<set<int>> pesos, string name="data"){
	int sensores = datos.size();
	int muestras = datos[0].size();
	//  Calculando serie temporal promedio
	vector<float> stPromedio(muestras);
	for(int m=0; m<muestras; m++){
		double acum = 0.0;
		for(int e=0; e<sensores; e++){
			acum += datos[e][m];
		}
		stPromedio[m] = (acum / sensores);
	}

	//  Marcando series temporales fijas: todos sus elementos iguales
	vector<bool> serieFija(sensores, true);
	for(int e=0; e<sensores; e++){
		for(int m=1; m<muestras; m++){
			serieFija[e] = serieFija[e] && (datos[e][m] == datos[e][m-1]);
		}
	}
	
	vector<double> zethaValues(sensores);
	for(int e=0; e<sensores; e++){
		zethaValues[e] = ZETA(datos[e], stPromedio);
	}

	// Cálculo de Medida de autocorrelación espacio-temporal
	int sumaW = 0;
	int cantFijas = 0;
	int cantSinVecinos = 0;
	int valorN = 0;
	double numerador = 0.0;
	double denominador = 0.0;
	for(int e=0; e<sensores; e++){
		if(!serieFija[e]){
			int vecinosValidos = 0;
			for(set<int>::iterator itVecs=pesos[e].begin(); itVecs != pesos[e].end(); itVecs++){
				double factor;
				if(!serieFija[*itVecs]){
					vecinosValidos++;
					sumaW += 1;
					factor = zethaValues[e] * zethaValues[*itVecs];
					numerador += factor;
				}
			}
			if(vecinosValidos>0){
				denominador += (zethaValues[e] * zethaValues[e]);
				valorN++;
			}else{
				cantSinVecinos++;
			}
		}else if(serieFija[e]){
			cantFijas++;
		}
	}

	// El valor N depende en este caso de los sensores considerados en la matriz de pesos
	int N = sensores - cantFijas - cantSinVecinos;
	cout << N << " (N) = " << sensores << " (sensores) - " << cantFijas << " (cantFijas) - " << cantSinVecinos << " (cantSinVecinos)" << endl;
	double IAET = ((0.0+N) / sumaW) * (numerador / denominador);
	cout << "N: " << N << " - valorN: " << valorN << endl;
	cout << setprecision(4) << fixed;
	cout << "dataset\t\tIAET\tN\ttotST\tlVecs\tstFijas\ttotalW" << endl;
	cout << name << "\t" << IAET << "\t" << N << "\t" << sensores << "\t" << pesos.size() << "\t" << cantFijas << "\t" << sumaW << endl;
}

int main(int argc, char const *argv[]){
	/*
		ESTA VERSIÓN MANEJA UN ATRIBUTO POR CADA ARCHIVO DE ENTRADA DE DATOS

		Se recibe como entrada las series de tiempo de los sensores de madrid con
		los datos a evaluar en el siguiente formato:
		num_sensores    num_muestras
		id.1 id.2 id.3 ... id.numsensores
		ts_id.1.1 ts_id.1.2 ... ts_id.1.nummuestras
		ts_id.2.1 ts_id.2.2 ... ts_id.2.nummuestras
		...
		ts_id.numsensores.1 ts_id.numsensores.2 ... ts_id.numsensores.nummuestras
		
		La lista de pesos debe estar en el siguiente formato:
		La primera línea la cantidad de sensores
		La segunda línea todos los ids de sensores
		Por cada sensor una línea indicando su identificador, cantidad de vecinos, identificador de cada vecino.

		Al cargar esta lista se crea una lista de vecindad simétrica entre cada par formado por el sensor y sus vecinos.
		Se descartan aquellos elementos que posean series de tiempo fijas (valores que no cambian)
	*/
	if(argc < 3){
		cout << "Error! Faltan argumentos." << endl;
		cout << argv[0] << " <time_series_sensors_madrid.txt> <w_list_file.txt> [<atribute_name>]" << endl;
		return 0;
	}
	string titulo = "data";
	if(argc > 3){
		titulo = argv[3];
	}
	cout << "DATASET: " << argv[1] << endl;
	ifstream dataSensores(argv[1], ifstream::in);
	if(dataSensores.fail()){
		cout << "Error! Lectura de " << argv[1] << " fallida." << endl;
		return -1;
	}
	int auxId;

	//				CARGANDO LISTA DE PESOS
	cout << "Lista de pesos W: " << argv[2] << endl;
	ifstream entradaListaPesos(argv[2], ifstream::in);
	if(entradaListaPesos.fail()){
		cout << "Error! Lectura de " << argv[2] << " fallida." << endl;
		return -1;
	}
	int cantIDs, cantVec, idVec, pos, posVec;
	entradaListaPesos >> cantIDs;
	cout << "Cantidad de Ids en la lista de pesos: " << cantIDs << endl;
	map<int, int> mapIdsPos;
	for(int i=0; i<cantIDs; i++){
		entradaListaPesos >> auxId;     // Identificador de la lista
		mapIdsPos[auxId] = i;
	}
	//cout << endl;
	vector<set<int>> listaPesos(cantIDs, set<int>());
	for(int i=0; i<cantIDs; i++){
		entradaListaPesos >> auxId;     // Identificador de la lista
		if(mapIdsPos.find(auxId) == mapIdsPos.end()){
			pos = -1;
		}else{
			pos = mapIdsPos[auxId];         // Posición en la lista de 
		}
		entradaListaPesos >> cantVec;    // Cantidad de elementos de la lista
		for(int j=0; j<cantVec; j++){
			entradaListaPesos >> idVec;
			//cout << idVec << " - ";
			if(pos >= 0 && mapIdsPos.find(idVec) != mapIdsPos.end()){
				posVec = mapIdsPos[idVec];
				listaPesos[pos].insert(posVec);
				listaPesos[posVec].insert(pos);
			}
		}
		//cout << endl;
	}
	entradaListaPesos.close();


	//		CARGANDO DATOS FILTRANDO SI EXISTE EN LA LISTA DE PESOS
	int sensores, muestras;
	dataSensores >> sensores >> muestras;
	cout << "Cantidad de sensores: " << sensores << endl;
	cout << "Cantidad de muestras: " << muestras << endl;
	vector<int> idsData(sensores);
	for(int i=0; i<sensores; i++){
		// Lectura de los identificadores de sensores
		dataSensores >> auxId;
		idsData[i] = auxId;
	}
	// Vectores con los datos
	vector<vector<int>> data(cantIDs, vector<int>(muestras));
	// Valores auxiliares para 'tapar' los valores nan
	string auxString;
	int nans = 0;
	for(int i=0; i<sensores; i++){
		auxId = idsData[i];
		if(mapIdsPos.find(auxId) != mapIdsPos.end()){
			pos = mapIdsPos[auxId];
			// Data
			for(int k=0; k<muestras; k++){
				dataSensores >> auxString;
				data[pos][k] = stoi(auxString);
			}
		}else{
			for(int k=0; k<muestras; k++){
				dataSensores >> auxString;
			}
		}
	}
	dataSensores.close();

	cout << "Cantidad sensores: " << data.size() << endl;
	cout << "Cantidad de nans: " << nans << endl;

	calculaIAET(data, listaPesos, titulo);
	return 0;
}
