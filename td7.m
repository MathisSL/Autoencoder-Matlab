clc;
clear;
close all;

% Chargement du dataset
dataset = load('emnist-digits.mat');

% Préparation des données d'entraînement
Xtrain = dataset.dataset.train.images(1:500, :)';
Xlabels = dataset.dataset.train.labels(1:500, :);
Xtrain_images = reshape(Xtrain, 28, 28, 1, []);

% Affichage d'une image avant reshape
figure;
imshow(reshape(Xtrain(:, 1), [28, 28]), []);
title('Première image');

% Affichage aléatoire de 20 images
num_images = 20;
max_start = size(Xtrain, 2) - num_images;
r = randi(max_start, 1);

figure;
for j = 1:num_images
    img = reshape(Xtrain(:, r + j - 1), [28, 28]);
    subplot(5, 4, j);
    imshow(img, []);
    title(['Label = ', num2str(Xlabels(r + j - 1))]);
end

% Encodage one-hot des labels d'entraînement
num_classes = 10;
one_hot_labels = zeros(num_classes, size(Xlabels, 1));
for i = 1:size(Xlabels, 1)
    one_hot_labels(Xlabels(i) + 1, i) = 1;
end

% Entraînement du premier autoencodeur
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(Xtrain, hiddenSize1, ...
    'MaxEpochs', 100, ...
    'ScaleData', false, ...
    'L2WeightRegularization', 0.004, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.15);

% Affichage des poids et de la structure
figure;
plotWeights(autoenc1);
view(autoenc1);

% Extraction des caractéristiques du premier encodeur
features1 = encode(autoenc1, Xtrain);

% Entraînement du second autoencodeur
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(features1, hiddenSize2, ...
    'MaxEpochs', 100, ...
    'ScaleData', false, ...
    'L2WeightRegularization', 0.004, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.15);

% Affichage des poids et de la structure
figure;
plotWeights(autoenc2);
view(autoenc2);

% Extraction des caractéristiques du second encodeur
features2 = encode(autoenc2, features1);

% Création et entraînement du réseau de neurones de classification
patNet = patternnet(num_classes, 'trainrp', 'crossentropy');
[patNet, tr] = train(patNet, features2, one_hot_labels);
view(patNet);

% Préparation des données de test
Xtest = dataset.dataset.test.images(1:500, :)';
Xtest_labels = dataset.dataset.test.labels(1:500, :);

% Encodage one-hot des labels de test
one_hot_labels_test = zeros(num_classes, size(Xtest_labels, 1));
for i = 1:size(Xtest_labels, 1)
    one_hot_labels_test(Xtest_labels(i) + 1, i) = 1;
end

% Empilement des autoencodeurs et du réseau de neurones
stacked_NN = stack(autoenc1, autoenc2, patNet);

% Prédictions sur le jeu de test
y = stacked_NN(Xtest);

% Conversion des prédictions en labels
[~, predicted_labels] = max(y, [], 1); % Trouver les indices des max
predicted_labels = predicted_labels - 1; % Ajuster pour correspondre aux indices des labels

% Affichage des prédictions et des labels réels
disp('Prédictions pour les 5 premières images :');
disp(predicted_labels(1:5));
disp('Labels réels pour les 5 premières images :');
disp(Xtest_labels(1:5)');

% Matrice de confusion
figure;
plotconfusion(one_hot_labels_test, y);

% Fine-tuning du réseau complet
stacked_NN = train(stacked_NN, Xtrain, one_hot_labels);

% Prédictions après fine-tuning
y_fine_tuned = stacked_NN(Xtest);
figure;
plotconfusion(one_hot_labels_test, y_fine_tuned);
