
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
	if (argc < 4) {
		cerr << "Error: faltan argumentos." << endl;
		cerr << argv[0] << " <input_file> <ids_sensores_m30.txt> <output_file>" << endl;
		return 1;
	}

	string archivoEntrada = argv[1];
	ifstream entrada(archivoEntrada);
	if (!entrada.is_open()) {
		cerr << "Error al abrir el archivo de entrada." << endl;
		return 1;
	}

	string archivoIdsM30 = argv[2];
	ifstream idsM30(archivoIdsM30);
	if (!idsM30.is_open()) {
		cerr << "Error al abrir el archivo de ids_sensores_m30.txt ." << endl;
		return 1;
	}
	vector<int> ids;
	int aux;
	while(idsM30 >> aux){
		ids.push_back(aux);
	}
	sort(ids.begin(), ids.end());
	cout << ids.size() << " ids de sensores de la M30 cargados." << endl;

	string archivoSalida = argv[3];
	ofstream salida(archivoSalida);
	if (!salida.is_open()) {
		cerr << "Error al crear el archivo de salida." << endl;
		return 1;
	}

	string linea;
	getline(entrada, linea);
	salida << linea << endl;
	int muestras = 0;
	while (getline(entrada, linea)) {
		aux = stoi(linea.substr(0, linea.find(';')));
		if (binary_search(ids.begin(), ids.end(), aux)) {
			salida << linea << endl;
			muestras++;
		}
	}
	cout << muestras << " muestras encontradas." << endl;

	entrada.close();
	salida.close();
	cout << "Archivo de salida creado correctamente." << endl;

	return 0;
}