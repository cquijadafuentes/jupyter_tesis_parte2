#include <iostream>
#include <sdsl/vectors.hpp>
#include <sdsl/bit_vectors.hpp>
#include <math.h>
#include <map>

using namespace std;
using namespace sdsl;

map<int,int> histoV0;
map<int,int> histoV1;
map<int,int> histoV2;
map<int,int> histoV3;
vector<int> minima(4, 0);
vector<int> maxima(4, 0);
vector<int> cantM(4,0);
vector<float> promedio(4, 0);
vector<float> varianza(4, 0);
vector<float> desvstandar(4, 0);
vector<long long int> acumulado(4, 0);

int golombM;
int golombB;
int golombT;

/*
g++ -std=c++11 -g -O0 -DNDEBUG -fopenmp -I ~/include -L ~/lib -o TS_explora_vlcvector_compara_series_doble_diferencia_todos TS_explora_vlcvector_compara_series_doble_diferencia_todos.cpp -lsdsl -ldivsufsort -ldivsufsort64
./TS_explora_vlcvector_compara_series_doble_diferencia_todos m30_2024_data_fulldates_intensidad.txt m30_2024_ids_estructura_intensidad.txt histograma_doble_diferencia_intensidad.txt
./TS_explora_vlcvector_compara_series_doble_diferencia_todos m30_2024_data_fulldates_ocupacion.txt m30_2024_ids_estructura_ocupacion.txt histograma_doble_diferencia_ocupacion.txt
./TS_explora_vlcvector_compara_series_doble_diferencia_todos m30_2024_data_fulldates_carga.txt m30_2024_ids_estructura_carga.txt histograma_doble_diferencia_carga.txt
./TS_explora_vlcvector_compara_series_doble_diferencia_todos m30_2024_data_fulldates_velocidad.txt m30_2024_ids_estructura_velocidad.txt histograma_doble_diferencia_velocidad.txt
*/

int zigzag_decode(int i){
	return ((i >> 1) ^ -(i & 1));
}

unsigned int zigzag_encode(int i){
	return ((i >> 31) ^ (i << 1));
}

int encuentraPosicion(vector<int> &l, int x){
	for(int i=0; i<l.size(); i++){
		if(l[i] == x){
			return i;
		}
	}
	return -1;
}

long long int size_bytes_golomb(int_vector<> x){
	// Estimación de bits
	long long int bits = 0;
	for(int n=0; n<x.size(); n++){
		int q = x[n] / golombM;
		int r = x[n] % golombM;
		bits += q+1;
		int b = golombB;
		if(r >= golombT){
			b++;
		}
		bits += b;
	}
	// Creación del bitvector
//	cout << "bits para bit_vector: " << bits << endl;
	bit_vector bvG = bit_vector(bits, 0);
	long long int p = 0;
	for(int n=0; n<x.size(); n++){
		int q = x[n] / golombM;
		int r = x[n] % golombM;
		for (int i = 0; i < q; i++) {
			bvG[p++] = 1;
		}
		bvG[p++] = 0;
		int b = golombB;
		if(r >= golombT){
			b++;
		}
		for(int i=b-1; i >= 0; i--){
			bvG[p+i] = (r >> (b - 1 - i)) & 1;
		}
		p += b;	
	}
	sd_vector<> sdb(bvG);
	return size_in_bytes(sdb);
}

void estadisticas();

vector<long long int> kilobytes_vectores_sin_zze(vector<int> &x){
	vector<long long int> r(5,0);
	int_vector<> ivBC(x.size());
	for(int i=0; i<x.size(); i++){
		ivBC[i] = x[i];
	}
	util::bit_compress(ivBC);
	r[0] = size_in_bytes(ivBC);
	vlc_vector<coder::elias_delta> vlcED(ivBC);
	r[1] = size_in_bytes(vlcED);
	vlc_vector<coder::elias_gamma> vlcEG(ivBC);
	r[2] = size_in_bytes(vlcEG);
	vlc_vector<coder::fibonacci> vlcFi(ivBC);
	r[3] = size_in_bytes(vlcFi);
	r[4] = size_bytes_golomb(ivBC);
	return r;
}

vector<long long int> kilobytes_vectores_con_zze(vector<int> &x){
	vector<long long int> r(5,0);
	int_vector<> ivBC(x.size());
	for(int i=0; i<x.size(); i++){
		ivBC[i] = zigzag_encode(x[i]);
	}
	util::bit_compress(ivBC);
	r[0] = size_in_bytes(ivBC);
	vlc_vector<coder::elias_delta> vlcED(ivBC);
	r[1] = size_in_bytes(vlcED);
	vlc_vector<coder::elias_gamma> vlcEG(ivBC);
	r[2] = size_in_bytes(vlcEG);
	vlc_vector<coder::fibonacci> vlcFi(ivBC);
	r[3] = size_in_bytes(vlcFi);
	r[4] = size_bytes_golomb(ivBC);
	return r;
}

vector<int> code_1(vector<int> &x){
	vector<int> c(x.size());
	c[0] = x[0];
	for(int i=1; i<x.size(); i++){
		c[i] = x[i] - x[i-1];
	}
	return c;
}

vector<int> code_2(vector<int> &x, vector<int> &ref){
	vector<int> c(x.size());
	for(int i=0; i<x.size(); i++){
		c[i] = x[i] - ref[i];
	}
	return c;
}

vector<int> code_2zz(vector<int> &x, vector<int> &ref){
	vector<int> c(x.size());
	for(int i=0; i<x.size(); i++){
		c[i] = zigzag_encode(x[i]) - zigzag_encode(ref[i]);
	}
	return c;
}

void agregaHisto(vector<int> &v, map<int,int> &m){
	for(int i=0; i<v.size(); i++){
		m[v[i]]++;
	}
}

int main(int argc, char const *argv[]){
	if(argc < 4){
		cout << "Error! Faltan argumentos." << endl;
		cout << argv[0] << " <inputFile> <groupsFile> <outputHistoFile> [<golomb_m>]" << endl;
		cout << "inputFile: archivo con los datos. Formato de atributo individual" << endl;
		cout << "groupsFile: archivo con los ids de los 5 grupos para la estructura." << endl;
		return 0;
	}
	cout << "Ejecutando:";
	for(int i=0; i<argc; i++){
		cout << " " << argv[i];
	}
	cout << endl;

	if(argc >= 5){
		golombM = stoi(argv[4]);
		golombB = log2(golombM);
		golombT = exp2(golombB+1) - golombM;
	}else{
		golombM = 10;
		golombB = log2(golombM);
		golombT = exp2(golombB+1) - golombM;
	}
	cout << "Valores golomb: " << endl;
	cout << "M: " << golombM << endl;
	cout << "B: " << golombB << endl;
	cout << "Tope: " << golombT << endl;

	// Leyendo datos desde el archivo de entrada	
	cout << "Iniciando lectura para codificación por semana" << endl;
	int aux, auxP;
	int cantGrupos = 5;
	ifstream listaIDS(argv[2], ifstream::in);
	if(listaIDS.fail()){
		cout << "Error! Lectura de " << argv[2] << " fallida." << endl;
		return -1;
	}
	vector<int> cantIds(cantGrupos);
	int totalIds = 0;
	for(int i=0; i<cantGrupos; i++){
		listaIDS >> cantIds[i];
		totalIds += cantIds[i];
	}
	cout << "Total Ids en grupos: " << totalIds << endl;
	map<int, pair<int,int>> mapeoIdsPos;
	vector<int> idsGroups(totalIds, 0);
	auxP = 0;
	for(int i=0; i<cantGrupos; i++){
		for(int j=0; j<cantIds[i]; j++){
			listaIDS >> aux;
			mapeoIdsPos[aux] = make_pair(i,j);
			idsGroups[auxP++] = aux;
		}
	}
	listaIDS.close();

	map<int, pair<int,int>>::iterator mIPit;
	int noEncontrado = 0;

	ifstream dataSensores(argv[1], ifstream::in);
	if(dataSensores.fail()){
		cout << "Error! Lectura de " << argv[1] << " fallida." << endl;
		return -1;
	}

	int sensores, muestras;
	dataSensores >> sensores >> muestras;
	cout << "Cantidad de sensores: " << sensores << endl;
	cout << "Cantidad de muestras: " << muestras << endl;

	if(sensores != totalIds){
		cout << "Lista de datos no coinciden: cantidad de IDs no coinciden." << endl;
	}

	vector<int> idsData(sensores);
	for(int i=0; i<sensores; i++){
		// Lectura de los identificadores de sensores
		dataSensores >> idsData[i];
		mIPit = mapeoIdsPos.find(idsData[i]);
		if(mIPit == mapeoIdsPos.end()){
			//cout << "Lista de datos no coinciden: ID en los datos no está agrupada." << endl;
			noEncontrado++;
		}
	}

	if(sensores - noEncontrado < totalIds){
		cout << "La cantidad de IDs en los datos es " << sensores << "." << endl;
		cout << "La cantidad de IDs en los grupos es " << totalIds << "." << endl;
		if(noEncontrado > 0){
			cout << noEncontrado << " IDs desde la fuente de datos que no se encontraron en los grupos." << endl;
		}
		if (totalIds < (sensores - noEncontrado)){
			cout << ((sensores - noEncontrado) - totalIds) << " IDs desde los grupos que no se encontraron en la fuente de datos." << endl;
		}
		
	}

	// Matriz con los datos
	vector<vector<vector<int>>> data(cantGrupos);
	for(int i=0; i<cantGrupos; i++){
		data[i] = vector<vector<int>>(cantIds[i], vector<int>(muestras));
	}
	pair<int,int> pPos;
	int posG, posI;
	for(int i=0; i<sensores; i++){
		mIPit = mapeoIdsPos.find(idsData[i]);
		if(mIPit != mapeoIdsPos.end()){
			pPos = mapeoIdsPos[idsData[i]];
			posG = pPos.first;
			posI = pPos.second;
			for(int k=0; k<muestras; k++){
				dataSensores >> aux;
				data[posG][posI][k] = aux;
			}
		}else{
			cout << "Error! no se encuentra en el mapa el id " << idsData[i] << endl;
			for(int k=0; k<muestras; k++){
				dataSensores >> aux;
			}
		}	
	}
	dataSensores.close();

	vector<long long int> bytesV0(5, 0);		// original	==>> bit_compress, elias_delta, elias_gamma, fibonacci, golomb
	vector<long long int> bytesV1(5, 0);		// code1	==>> bit_compress, elias_delta, elias_gamma, fibonacci, golomb
	vector<long long int> bytesV2(5, 0);		// code2	==>> bit_compress, elias_delta, elias_gamma, fibonacci, golomb
	vector<long long int> bytesV3(5, 0);		// code2zz	==>> bit_compress, elias_delta, elias_gamma, fibonacci, golomb
	
	for(int f=0; f<data.size(); f++){
		//	Serie de referencia del grupo (la primera)
		//	Tamaño datos originales
		vector<long long int> tbytes = kilobytes_vectores_sin_zze(data[f][0]);
		for(int i=0; i<4; i++){
			bytesV0[i] += tbytes[i];
		}
		//	Referencia en code1 (se suma al tamaño de code2 y code2zz)
		vector<int> refC1 = code_1(data[f][0]);
		tbytes = kilobytes_vectores_con_zze(refC1);
		for(int i=0; i<4; i++){
			bytesV1[i] += tbytes[i];
			bytesV2[i] += tbytes[i];
			bytesV3[i] += tbytes[i];
		}
		//	Histograma
		agregaHisto(data[f][0], histoV0);
		agregaHisto(refC1, histoV1);
		agregaHisto(refC1, histoV2);
		agregaHisto(refC1, histoV3);
		for(int c=1; c<data[f].size(); c++){
			//	Tamaño de la serie original
			tbytes = kilobytes_vectores_sin_zze(data[f][c]);
			for(int i=0; i<tbytes.size(); i++){
				bytesV0[i] += tbytes[i];
			}
			//	Tamaño de la serie code 1
			vector<int> c1 = code_1(data[f][c]);
			tbytes = kilobytes_vectores_con_zze(c1);
			for(int i=0; i<tbytes.size(); i++){
				bytesV1[i] += tbytes[i];
			}
			//	Tamaño de la serie code 2
			vector<int> c2 = code_2(c1, refC1);
			tbytes = kilobytes_vectores_con_zze(c2);
			for(int i=0; i<tbytes.size(); i++){
				bytesV2[i] += tbytes[i];
			}
			//	Tamaño de la serie code 2zz
			vector<int> c3 = code_2zz(c1, refC1);
			tbytes = kilobytes_vectores_con_zze(c3);
			for(int i=0; i<tbytes.size(); i++){
				bytesV3[i] += tbytes[i];
			}
			//	Histograma
			agregaHisto(data[f][c], histoV0);
			agregaHisto(c1, histoV1);
			agregaHisto(c2, histoV2);
			agregaHisto(c3, histoV3);
		}
	}
	cout << " ********************** " << endl;
	cout << "\t\tBC\tED\tEG\tFi\tGo" << endl;
	cout << "Datos_Original";
	for(int i=0; i<bytesV0.size(); i++){
		cout << "\t" << bytesV0[i];
	}
	cout << "\t[Bytes]" << endl;
	cout << "Datos_Code1";
	for(int i=0; i<bytesV1.size(); i++){
		cout << "\t" << bytesV1[i];
	}
	cout << "\t[Bytes]" << endl;
	cout << "Datos_Code2";
	for(int i=0; i<bytesV2.size(); i++){
		cout << "\t" << bytesV2[i];
	}
	cout << "\t[Bytes]" << endl;
	cout << "D_Code2zz";
	for(int i=0; i<bytesV3.size(); i++){
		cout << "\t" << bytesV3[i];
	}
	cout << "\t[Bytes]" << endl;
	cout << " ********************** " << endl;
	cout << "\t\tBC\tED\tEG\tFi\tGo" << endl;
	cout << "Datos_Original";
	for(int i=0; i<bytesV0.size(); i++){
		cout << "\t" << (bytesV0[i] / 1024);
	}
	cout << "\t[KiloBytes]" << endl;
	cout << "Datos_Code1";
	for(int i=0; i<bytesV1.size(); i++){
		cout << "\t" << (bytesV1[i] / 1024);
	}
	cout << "\t[KiloBytes]" << endl;
	cout << "Datos_Code2";
	for(int i=0; i<bytesV2.size(); i++){
		cout << "\t" << (bytesV2[i] / 1024);
	}
	cout << "\t[KiloBytes]" << endl;
	cout << "D_Code2zz";
	for(int i=0; i<bytesV3.size(); i++){
		cout << "\t" << (bytesV3[i] / 1024);
	}
	cout << "\t[KiloBytes]" << endl;
	cout << " ********************** " << endl;

	estadisticas();

	int min_01 = (minima[0] < minima[1]) ? minima[0] : minima[1];
	int min_23 = (minima[2] < minima[3]) ? minima[2] : minima[3];
	int minmin = (min_01 < min_23) ? min_01 : min_23;

	int max_01 = (maxima[0] > maxima[1]) ? maxima[0] : maxima[1];
	int max_23 = (maxima[2] > maxima[3]) ? maxima[2] : maxima[3];
	int maxmax = (max_01 > max_23) ? max_01 : max_23;

	ofstream histograma(argv[3]);
	cout << "Creando histograma desde valor " << minmin << " hasta valor " << maxmax  << " ..." << endl;
	histograma << "% valor\tNoC_0\tNoC_1\tNoC_2\tNoC_3" << endl;
	for(int i=minmin; i<maxmax; i++){
		histograma << i << "\t" << histoV0[i] << "\t" << histoV1[i] << "\t" << histoV2[i] << "\t" << histoV3[i] << endl;
	}
	histograma.close();
	cout << "Histograma creado con éxito." << endl;
	return 0;
}

void estadisticas(){
	// ************** Minimo, maximo, acumulado y promedio **************
	int aux;
	// ------- V0 -------
	minima[0] = (histoV0.begin())->first;
	maxima[0] = histoV0.begin()->first;
	for(auto it = histoV0.begin(); it != histoV0.end(); it++){
		aux = int(it->first * it->second);
		acumulado[0] += aux;
		cantM[0] += it->second;
		if(it->first < minima[0]){
			minima[0] = it->first;
		}
		if(it->first > maxima[0]){
			maxima[0] = it->first;
		}
	}
	promedio[0] = (0.0 + acumulado[0]) / cantM[0];
	// ------- V1 -------
	minima[1] = (histoV1.begin())->first;
	maxima[1] = histoV1.begin()->first;
	for(auto it = histoV1.begin(); it != histoV1.end(); it++){
		aux = (it->first * it->second);
		acumulado[1] += aux;
		cantM[1] += it->second;
		if(it->first < minima[1]){
			minima[1] = it->first;
		}
		if(it->first > maxima[1]){
			maxima[1] = it->first;
		}
	}
	promedio[1] = (0.0 + acumulado[1]) / cantM[1];
	// ------- V2 -------
	minima[2] = (histoV2.begin())->first;
	maxima[2] = histoV2.begin()->first;
	for(auto it = histoV2.begin(); it != histoV2.end(); it++){
		aux = (it->first * it->second);
		acumulado[2] += aux;
		cantM[2] += it->second;
		if(it->first < minima[2]){
			minima[2] = it->first;
		}
		if(it->first > maxima[2]){
			maxima[2] = it->first;
		}
	}
	promedio[2] = (0.0 + acumulado[2]) / cantM[2];
	// ------- V3 -------
	minima[3] = (histoV3.begin())->first;
	maxima[3] = histoV3.begin()->first;
	for(auto it = histoV3.begin(); it != histoV3.end(); it++){
		aux = (it->first * it->second);
		acumulado[3] += aux;
		cantM[3] += it->second;
		if(it->first < minima[3]){
			minima[3] = it->first;
		}
		if(it->first > maxima[3]){
			maxima[3] = it->first;
		}
	}
	promedio[3] = (0.0 + acumulado[3]) / cantM[3];
	// ************** Varianza y Desviación Estándar **************
	vector<long long int> acumDE(4,0);
	// ------- V0 -------
	for(auto it = histoV0.begin(); it != histoV0.end(); it++){
		aux = (it->first - promedio[0]);
		acumDE[0] += aux * aux * it->second;
	}
	varianza[0] = acumDE[0] / cantM[0];
	desvstandar[0] = sqrt(varianza[0]);
	// ------- V1 -------
	for(auto it = histoV1.begin(); it != histoV1.end(); it++){
		aux = (it->first - promedio[1]);
		acumDE[1] += aux * aux * it->second;
	}
	varianza[1] = acumDE[1] / cantM[1];
	desvstandar[1] = sqrt(varianza[1]);
	// ------- V2 -------
	for(auto it = histoV2.begin(); it != histoV2.end(); it++){
		aux = (it->first - promedio[2]);
		acumDE[2] += aux * aux * it->second;
	}
	varianza[2] = acumDE[2] / cantM[2];
	desvstandar[2] = sqrt(varianza[2]);
	// ------- V3 -------
	for(auto it = histoV3.begin(); it != histoV3.end(); it++){
		aux = (it->first - promedio[3]);
		acumDE[3] += aux * aux * it->second;
	}
	varianza[3] = acumDE[3] / cantM[3];
	desvstandar[3] = sqrt(varianza[3]);

	// ************** Mostrando **************

	cout << "\tOrig.\tCode 1\tCode 2\tCode 2zz" << endl;
	cout << "Mínima\t" << minima[0] << "\t" << minima[1] << "\t" << minima[2] << "\t" << minima[3] << endl;
	cout << "Máxima\t" << maxima[0] << "\t" << maxima[1] << "\t" << maxima[2] << "\t" << maxima[3] << endl;
	cout << "Media\t" << promedio[0] << "\t" << promedio[1] << "\t" << promedio[2] << "\t" << promedio[3] << endl;
	cout << "Total\t" << acumulado[0] << "\t" << acumulado[1] << "\t" << acumulado[2] << "\t" << acumulado[3] << endl;
	cout << "NumMues\t" << cantM[0] << "\t" << cantM[1] << "\t" << cantM[2] << "\t" << cantM[3] << endl;
	cout << fixed;
    cout << setprecision(2);
	cout << "Varian\t" << varianza[0] << "\t" << varianza[1] << "\t" << varianza[2] << "\t" << varianza[3] << endl;
	cout << "DesEst\t" << desvstandar[0] << "\t" << desvstandar[1] << "\t" << desvstandar[2] << "\t" << desvstandar[3] << endl;
	cout << " ********************** " << endl;

}
