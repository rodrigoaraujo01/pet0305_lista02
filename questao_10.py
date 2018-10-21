import numpy as np

x = np.matrix([[79.9, 78.5, 68.9, 62.2, 69.2, 67.8, 61.3, 71.6, 83.7, 67.1, 59.8, 66.7, 72.8, 60.9, 61.4, 75.0, 80.5, 71.3, 56.6, 55.9, 61.5, 59.2, 76.9, 58.0],
               [13.9, 16.3, 22.6, 20.2, 23.7, 19.8, 24.9, 19.2, 10.5, 26.5, 27.9, 23.2, 14.5, 28.9, 29.2, 16.8, 11.9, 18.5, 28.9, 32.8, 28.1, 28.4, 16.3, 27.6],
               [6.2, 7.2, 8.5, 17.6, 7.1, 12.4, 13.8, 9.2, 5.8, 6.4, 12.3, 10.1, 12.7, 10.2, 9.4, 8.2, 7.6, 10.2, 14.5, 11.3, 10.4, 12.4, 6.8, 14.4],
               [3.3, 2.5, 3.6, 2.8, 0.9, 3.8, 2.2, 3.6, 4.4, 1.4, 3.5, 2.9, 1.9, 1.5, 2.5, 3.1, 3.8, 2.6, 2.8, 3.1, 2.7, 2.8, 2.9, 3.4]])

for i, row in enumerate(x):
    print(f'X{i+1}')
    tmp = np.array(row)
    print(f'Média: {np.average(tmp)}')
    print(f'Mediana: {np.median(tmp)}')
    print(f'Desvio padrão: {np.std(tmp)}')
    print(f'Máximo: {np.max(tmp)}')
    print(f'Mínimo: {np.min(tmp)}')
    print('\n')

# Compute covariance matrix
cov = np.cov(x)
print('Matriz de covariância')
print(cov)

# Compute eigen_values and eigen_vectors
eigen_values, eigen_vectors = np.linalg.eig(cov)

# print('\n',eigen_vectors,'\n')

# print(np.dot(cov, eigen_vectors[:,0]))
# print(eigen_values[0] * eigen_vectors[:,0])

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\nEigen values - Eigen vectors')
for i in eigen_pairs:
    print(i[0], ' - ', i[1])

for i, (eva, eve) in enumerate(eigen_pairs):
    aux = f'Y{i+1} = '
    for j, comp in enumerate(eve):
        aux += f'{eigen_pairs[j][1][i]:0.2f} X{j+1} + '
    print(aux)

for eva, eve in eigen_pairs:
    print(eva/sum([x[0] for x in eigen_pairs]))