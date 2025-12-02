#include <iostream>

// Read a 2x2 matrix from input and display it
int main()
{
	int mat[2][2];

	std::cout << "Enter 4 integers for a 2x2 matrix (row-wise):\n";
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			if (!(std::cin >> mat[i][j])) {
				std::cerr << "Error: failed to read input\n";
				return 1;
			}
		}
	}

	std::cout << "Matrix:\n";
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			std::cout << mat[i][j] << ' ';
		}
		std::cout << '\n';
	}

	return 0;
}