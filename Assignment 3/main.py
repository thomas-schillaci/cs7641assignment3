import io
import time
import warnings
from decimal import Decimal

import matplotlib.cm as cm
import matplotlib.pyplot as plot
import numpy as np
import pydotplus
import utils
from keras.utils import to_categorical
from prince import MCA
from scipy.stats import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def density_categorical_accuracy(labels, predicted_labels, classes):
    assert (len(labels) == len(predicted_labels))
    if len(labels) == 0:
        return 0

    n_cluster = np.max(predicted_labels) + 1
    clusters = [[] for _ in range(n_cluster)]

    for label, predicted_label in zip(labels, predicted_labels):
        clusters[predicted_label].append(label)

    catacc = np.average([stats.mode(d)[1][0] / len(d) for d in clusters], weights=[len(d) for d in clusters])
    corrected_catacc = (catacc - 1.0 / classes) / (1.0 - 1.0 / classes)

    return corrected_catacc


def silhouette(name, n_clusters, x):
    averages = []

    for n_cluster in n_clusters:
        plot.style.use('seaborn-darkgrid')
        plot.title(f'Silhouette on the {name} dataset, using {n_cluster}-means')
        ax = plot.gca()

        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(x) + (n_cluster + 1) * 10])

        clusterer = KMeans(n_clusters=n_cluster, random_state=0)
        cluster_labels = clusterer.fit_predict(x)

        silhouette_avg = silhouette_score(x, cluster_labels)
        averages.append(silhouette_avg)

        sample_silhouette_values = silhouette_samples(x, cluster_labels)

        y_lower = 10
        for i in range(n_cluster):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10

        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster labels")

        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plot.show()

    lb = np.min(averages)
    ub = np.max(averages)
    amplitude = ub - lb
    lb -= 0.2 * amplitude
    ub += 0.2 * amplitude

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Silhouette averages on the {name} dataset using k-means')
    plot.bar(n_clusters, averages)
    plot.xticks(n_clusters)
    plot.xlabel('Number of clusters')
    plot.ylabel('Silhouette averages')
    plot.ylim([lb, ub])
    plot.show()

    print(f'{name}: {averages}')


def cataccs(name, n_clusters, classes, x, y):
    res = []
    for n_cluster in n_clusters:
        clf = KMeans(n_clusters=n_cluster, random_state=0)
        pred = clf.fit_predict(x)
        catacc = density_categorical_accuracy(y, pred, classes) * 100
        res.append(catacc)

    lb = np.min(res)
    ub = np.max(res)
    amplitude = ub - lb
    lb -= 0.2 * amplitude
    ub += 0.2 * amplitude

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Influence of k on the categorical accuracy on {name}')
    plot.bar(n_clusters, res)
    plot.xticks(n_clusters)
    plot.xlabel('Number of clusters')
    plot.ylabel('Categorical accuracy (%)')
    plot.ylim([lb, ub])
    plot.show()


def cluster_breakdown(name, x, predicted_labels):
    clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=5)
    clf.fit(x, predicted_labels)
    dot_data = io.StringIO()
    export_graphviz(clf, out_file=dot_data, feature_names=x.keys())
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(f'{name}_bd_tree.pdf')


def expectation_maximization_silhouettes(name, x):
    averages = []
    n_clusters = range(2, 15)

    for k in n_clusters:
        clf = GaussianMixture(n_components=k)
        predicted_labels = clf.fit_predict(x)
        silhouette_avg = silhouette_score(x, predicted_labels)
        averages.append(silhouette_avg)

    lb = np.min(averages)
    ub = np.max(averages)
    amplitude = ub - lb
    lb -= 0.2 * amplitude
    ub += 0.2 * amplitude

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Silhouette averages on the {name} dataset using EM')
    plot.bar(n_clusters, averages)
    plot.xticks(n_clusters)
    plot.xlabel('Number of clusters')
    plot.ylabel('Silhouette averages')
    plot.ylim([lb, ub])
    plot.show()


def expectation_maximization_thresholds(x, y, k, powers=range(14)):
    thresholds = [1 - Decimal(str(f'1e-{k}')) for k in powers]
    samples = [[] for _ in range(len(thresholds))]
    trials = 3

    for _ in range(trials):
        clf = GaussianMixture(n_components=k)
        clf.fit(x)
        probabilities = clf.predict_proba(x)

        for i, threshold in enumerate(thresholds):
            labels = []
            predicted_labels = []
            for j, probability in enumerate(probabilities):
                argmax = np.argmax(probability)
                if probability[argmax] >= threshold:
                    labels.append(y[j])
                    predicted_labels.append(argmax)
            catacc = density_categorical_accuracy(labels, predicted_labels, k) * 100
            samples[i].append(catacc)

    cataccs = np.median(samples, axis=1)
    cataccs_std = np.std(samples, axis=1)

    plot.style.use('seaborn-darkgrid')
    plot.title('Influence of the probability threshold on the categorical accuracy')
    plot.xticks(powers, [f'1-1e-{k}' for k in powers], rotation=60)
    plot.xlabel('Probability threshold')
    plot.ylabel('Categorical accuracy')
    plot.fill_between(powers, cataccs - cataccs_std / 2, cataccs + cataccs_std / 2, alpha=0.5)
    plot.plot(powers, cataccs, 'o-')
    plot.show()


def pca(name, x, y):
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(x)

    plot.style.use('seaborn-darkgrid')
    plot.title(f'PCA on {name}')
    plot.xlabel('First dimension')
    plot.ylabel('Second dimension')
    plot.scatter(transformed[:, 0], transformed[:, 1], c=y, cmap='viridis')
    plot.show()


def pca_benchmark(name, x, clf):
    averages = []
    n_components = range(2, 15)

    for n_component in n_components:
        pca = PCA(n_components=n_component)
        transformed = pca.fit_transform(x)
        predicted_labels = clf.fit_predict(transformed)
        silhouette_avg = silhouette_score(transformed, predicted_labels)
        averages.append(silhouette_avg)

    lb = np.min(averages)
    ub = np.max(averages)
    amplitude = ub - lb
    lb -= 0.2 * amplitude
    ub += 0.2 * amplitude

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Silhouette averages on the {name} dataset using {repr(clf).split("(")[0]} and PCA')
    plot.bar(n_components, averages)
    plot.xticks(n_components)
    plot.xlabel('Number of components')
    plot.ylabel('Silhouette averages')
    plot.ylim([lb, ub])
    plot.show()


def mca(name, x, y):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    ma = MCA(n_components=2)
    transformed = ma.fit_transform(x).values

    plot.style.use('seaborn-darkgrid')
    plot.title(f'MCA on {name}')
    plot.xlabel('First dimension')
    plot.ylabel('Second dimension')
    plot.scatter(transformed[:, 0], transformed[:, 1], c=y, cmap='viridis')
    plot.show()


def mca_benchmark(name, x, clf):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    averages = []
    n_components = range(1, 5)

    for n_component in n_components:
        mca = MCA(n_components=n_component)
        transformed = mca.fit_transform(x).values
        predicted_labels = clf.fit_predict(transformed)
        silhouette_avg = silhouette_score(transformed, predicted_labels)
        averages.append(silhouette_avg)

    lb = np.min(averages)
    ub = np.max(averages)
    amplitude = ub - lb
    lb -= 0.2 * amplitude
    ub += 0.2 * amplitude

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Silhouette averages on the {name} dataset using {repr(clf).split("(")[0]} and MCA')
    plot.bar(n_components, averages)
    plot.xticks(n_components)
    plot.xlabel('Number of components')
    plot.ylabel('Silhouette averages')
    plot.ylim([lb, ub])
    plot.show()


def pca_eigenvalues(x_adult, x_wine):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pca = PCA(n_components=10)
    pca.fit(x_adult)
    y_adult = pca.explained_variance_

    pca = PCA(n_components=10)
    pca.fit(x_wine)
    y_wine = pca.explained_variance_

    mca = MCA(n_components=10)
    mca.fit(x_wine)
    y_wine2 = 100 * np.array(mca.eigenvalues_)

    x_axis = [k + 1 for k in range(10)]

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Eigen values distributions')
    plot.xlabel('Eigen value index')
    plot.ylabel('Eigen value')
    plot.xticks(x_axis, x_axis)
    plot.plot(x_axis, np.transpose([y_adult, y_wine, y_wine2]), 'o-')
    plot.legend(['Adult', 'Wine reviews (PCA)', 'Wine reviews (MCA) x100'], loc='upper right')
    plot.show()


def ica(name, x, y):
    ica = FastICA(n_components=2)
    transformed = ica.fit_transform(x)

    plot.style.use('seaborn-darkgrid')
    plot.title(f'ICA on {name}')
    plot.xlabel('First dimension')
    plot.ylabel('Second dimension')
    plot.scatter(transformed[:, 0], transformed[:, 1], c=y, cmap='viridis')
    plot.show()


def ica_benchmark(name, x, clf):
    averages = []
    n_components = range(2, 15)

    for n_component in n_components:
        ica = FastICA(n_components=n_component)
        transformed = ica.fit_transform(x)
        predicted_labels = clf.fit_predict(transformed)
        silhouette_avg = silhouette_score(transformed, predicted_labels)
        averages.append(silhouette_avg)

    lb = np.min(averages)
    ub = np.max(averages)
    amplitude = ub - lb
    lb -= 0.2 * amplitude
    ub += 0.2 * amplitude

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Silhouette averages on the {name} dataset using {repr(clf).split("(")[0]} and ICA')
    plot.bar(n_components, averages)
    plot.xticks(n_components)
    plot.xlabel('Number of components')
    plot.ylabel('Silhouette averages')
    plot.ylim([lb, ub])
    plot.show()


def kurtosis(name, x, n_components):
    ica = FastICA(n_components=n_components)
    transformed = ica.fit_transform(x)

    print(f'Kurtosis of the {name} dataset on {n_components} components:')
    print(stats.kurtosis(transformed))


def ica_projection(name, x, n_components):
    ica = FastICA(n_components=n_components)
    ica.fit(x)

    unmixing = ica.components_

    for component in range(n_components):
        plot.bar([k for k in range(len(unmixing[0]))], unmixing[component], width=2)
        plot.show()

    for component in range(n_components):
        argmaxes = (-np.abs(unmixing[component])).argsort()[:8]
        print(f'8 most important features of the {name} dataset after ICA along the component #{component}:')
        for argmax in argmaxes:
            print(x.keys()[argmax])


def jlmd_search(ubs, names):
    epsilons = np.linspace(0.2, 0.999, 1000)
    y = []

    for eps in epsilons:
        y.append(johnson_lindenstrauss_min_dim(40000, eps))

    plot.style.use('seaborn-darkgrid')
    ax = plot.subplots()[1]
    plot.title('Influence of epsilon on the minimum number of dimensions')
    plot.semilogy(epsilons, y)
    for ub in ubs:
        plot.semilogy([0, 1], [ub, ub])
    plot.legend(['Minimum number of dimensions', *names], loc='upper right')
    plot.show()


def rp(name, x, y):
    plot.style.use('seaborn-darkgrid')

    for i in range(6):
        rp = GaussianRandomProjection(eps=0.95, random_state=i)
        transformed = rp.fit_transform(x)

        axes = [0, 0]
        axes_std = [0, 0]

        for axis in range(np.shape(transformed)[1]):
            std = np.std(transformed[:, axis])
            if std > axes_std[0]:
                axes[0] = axis
                axes_std[0] = std
            elif std > axes_std[1]:
                axes[1] = axis
                axes_std[1] = std

        plot.subplot(2, 3, i + 1)
        plot.title(f'Random seed = {i}')
        plot.xlabel(f'Dimension {axes[0]}')
        plot.ylabel(f'Dimension {axes[1]}')
        plot.scatter(transformed[:, axes[0]], transformed[:, axes[1]], c=y, cmap='viridis')

    plot.show()


def rp_benchmark(name, x, clf):
    epsilons = np.linspace(0.35, 0.95, 7)
    n_components = []
    averages = []
    averages_err = [[], []]
    trials = 4

    for eps in epsilons:
        samples = []
        transformed = None
        for _ in range(trials):
            rp = GaussianRandomProjection(random_state=0, eps=eps)
            transformed = rp.fit_transform(x)
            predict = clf.fit_predict(transformed)
            silhouette_avg = silhouette_score(transformed, predict)
            samples.append(silhouette_avg)
        n_components.append(len(transformed[0]))
        mean = np.mean(samples)
        averages.append(mean)
        averages_err[0].append(mean - np.min(samples))
        averages_err[1].append(np.max(samples) - mean)

    lb = np.min(averages)
    ub = np.max(averages)
    amplitude = ub - lb
    lb -= 0.2 * amplitude
    ub += 0.2 * amplitude

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Silhouette averages using Random Projection on {name} using {repr(clf).split("(")[0]}')
    plot.bar([k for k in range(len(averages))], averages, yerr=averages_err)
    plot.xticks([k for k in range(len(averages))], n_components)
    plot.xlabel('Number of components')
    plot.ylabel('Silhouette averages')
    plot.ylim([lb, ub])
    plot.show()


def vae(name, x, y):
    encoder, vae = utils.create_vae(np.shape(x)[1])
    vae.fit(x, batch_size=50, epochs=10)
    transformed = encoder.predict(x)

    plot.style.use('seaborn-darkgrid')
    plot.title(f'VAE on {name}')
    plot.xlabel('First dimension')
    plot.ylabel('Second dimension')
    plot.scatter(transformed[:, 0], transformed[:, 1], c=y, cmap='viridis')
    plot.show()


def vae_benchmark(name, x, clf):
    averages = []
    latent_dims = range(2, 10)

    for latent_dim in latent_dims:
        encoder, vae = utils.create_vae(np.shape(x)[1], latent_dim=latent_dim)
        vae.fit(x, batch_size=50, epochs=5)
        transformed = encoder.predict(x)
        predicted_labels = clf.fit_predict(transformed)
        silhouette_avg = silhouette_score(transformed, predicted_labels)
        averages.append(silhouette_avg)

    lb = np.min(averages)
    ub = np.max(averages)
    amplitude = ub - lb
    lb -= 0.2 * amplitude
    ub += 0.2 * amplitude

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Silhouette averages on the {name} dataset using {repr(clf).split("(")[0]} and VAE')
    plot.bar(latent_dims, averages)
    plot.xticks(latent_dims)
    plot.xlabel('Number of components')
    plot.ylabel('Silhouette averages')
    plot.ylim([lb, ub])
    plot.show()


def reduction_clustering(xs, ys, classes, clfs, pcas, icas, rps, encoders_vaes):
    cataccs = []

    for i in range(2):
        predicted = clfs[i].fit_predict(xs[i])
        cataccs.append(density_categorical_accuracy(ys[i], predicted, classes[i]) * 100)

    for reducers in [pcas, icas, rps]:
        for i in range(2):
            reducer = reducers[i]
            if reducer is None:
                cataccs.append(0)
                continue
            transformed = reducer.fit_transform(xs[i])
            predicted = clfs[i].fit_predict(transformed)
            cataccs.append(density_categorical_accuracy(ys[i], predicted, classes[i]) * 100)

    for i in range(2):
        encoders_vaes[i][1].fit(xs[i], batch_size=50, epochs=10)
        transformed = encoders_vaes[i][0].predict(xs[i])
        predicted = clfs[i].fit_predict(transformed)
        cataccs.append(density_categorical_accuracy(ys[i], predicted, classes[i]) * 100)

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Influence of feature transformation on the categorical accuracy')
    color = []
    for _ in range(5):
        color.append('tab:blue')
        color.append('tab:orange')
    x = []
    count = 1
    for _ in range(5):
        x.append(count)
        count += 0.5
        x.append(count)
        count += 1
    plot.bar(x, cataccs, color=color, width=0.75)
    x = []
    count = 1.25
    for _ in range(5):
        x.append(count)
        count += 1.5
    plot.xticks(x, ['None', 'PCA', 'ICA', 'RP', 'VAE'])
    plot.xlabel('Feature transformation method')
    plot.ylabel('Categorical accuracy (%)')
    plot.show()


def nn(xs, ys, xs_test, ys_test):
    n_components = [0 for _ in range(10)]
    cataccs = [0 for _ in range(10)]

    ys = [to_categorical(ys[0]), to_categorical(ys[1])]
    ys_test = [to_categorical(ys_test[0]), to_categorical(ys_test[1])]

    for i in range(2):
        shape = np.shape(xs[i])[1]
        n_components[i] = shape
        model = utils.create_adult_model(shape, 2) if i == 0 else utils.create_wine_model(shape, 5)
        model.fit(xs[i][:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
        cataccs[i] = model.evaluate(xs_test[i], ys_test[i], verbose=False)[1] * 100

    for k in range(2, 11):
        for i in range(2):
            pca = PCA(n_components=k)
            transformed = pca.fit_transform(xs[i])
            transformed_test = pca.transform(xs_test[i])
            model = utils.create_adult_model(k, 2) if i == 0 else utils.create_wine_model(k, 5)
            model.fit(transformed[:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            catacc = model.evaluate(transformed_test, ys_test[i], verbose=False)[1] * 100
            if catacc > cataccs[2 + i]:
                n_components[2 + i] = k
                cataccs[2 + i] = catacc

            ica = FastICA(n_components=k)
            transformed = ica.fit_transform(xs[i])
            transformed_test = ica.transform(xs_test[i])
            model = utils.create_adult_model(k, 2) if i == 0 else utils.create_wine_model(k, 5)
            model.fit(transformed[:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            catacc = model.evaluate(transformed_test, ys_test[i], verbose=False)[1] * 100
            if catacc > cataccs[4 + i]:
                n_components[4 + i] = k
                cataccs[4 + i] = catacc

            if i == 1 and cataccs[6] == 0:
                rp = GaussianRandomProjection(eps=0.95)
                transformed = rp.fit_transform(xs[i])
                dim = np.shape(transformed)[1]
                transformed_test = rp.transform(xs_test[i])
                model = utils.create_wine_model(dim, 5)
                model.fit(transformed[:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
                catacc = model.evaluate(transformed_test, ys_test[i], verbose=False)[1] * 100
                if catacc > cataccs[6 + i]:
                    n_components[6 + i] = k
                    cataccs[6 + i] = catacc

            encoder, vae = utils.create_vae(np.shape(xs[i])[1], k)
            vae.fit(xs[i], batch_size=50, epochs=10, verbose=False)
            transformed = encoder.predict(xs[i], verbose=False)
            transformed_test = encoder.predict(xs_test[i], verbose=False)
            model = utils.create_adult_model(k, 2) if i == 0 else utils.create_wine_model(k, 5)
            model.fit(transformed[:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            catacc = model.evaluate(transformed_test, ys_test[i], verbose=False)[1] * 100
            if catacc > cataccs[8 + i]:
                n_components[8 + i] = k
                cataccs[8 + i] = catacc

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Influence of feature transformation on the NN accuracy')
    color = []
    for _ in range(5):
        color.append('tab:blue')
        color.append('tab:orange')
    x = []
    count = 1
    for _ in range(5):
        x.append(count)
        count += 0.5
        x.append(count)
        count += 1
    plot.bar(x, cataccs, color=color, width=0.75)
    x = []
    count = 1.25
    for _ in range(5):
        x.append(count)
        count += 1.5
    plot.xticks(x, ['None', 'PCA', 'ICA', 'RP', 'VAE'])
    plot.xlabel('Feature transformation method')
    plot.ylabel('Categorical accuracy (%)')
    plot.show()


def nn_benchmark(xs, ys, n_components):
    ys = [to_categorical(ys[0]), to_categorical(ys[1])]

    none_samples = [[], []]
    pca_samples = [[], []]
    ica_samples = [[], []]
    rp_samples = [[], []]
    vae_samples = [[], []]

    trials = 7
    for _ in range(trials):

        for i in range(2):
            shape = np.shape(xs[i])[1]
            n_components[i] = shape
            model = utils.create_adult_model(shape, 2) if i == 0 else utils.create_wine_model(shape, 5)
            start = time.time()
            model.fit(xs[i][:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            none_samples[i].append(time.time() - start)

        for i in range(2):
            dim = n_components[2 + i]
            pca = PCA(n_components=dim)
            transformed = pca.fit_transform(xs[i])
            model = utils.create_adult_model(dim, 2) if i == 0 else utils.create_wine_model(dim, 5)
            start = time.time()
            model.fit(transformed[:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            pca_samples[i].append(time.time() - start)

            dim = n_components[4 + i]
            ica = FastICA(n_components=dim)
            transformed = ica.fit_transform(xs[i])
            model = utils.create_adult_model(dim, 2) if i == 0 else utils.create_wine_model(dim, 5)
            start = time.time()
            model.fit(transformed[:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            ica_samples[i].append(time.time() - start)

            if i == 1:
                rp = GaussianRandomProjection(eps=0.95)
                transformed = rp.fit_transform(xs[i])
                dim = np.shape(transformed)[1]
                model = utils.create_wine_model(dim, 5)
                start = time.time()
                model.fit(transformed[:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
                rp_samples[i].append(time.time() - start)

            dim = n_components[8 + i]
            encoder, vae = utils.create_vae(np.shape(xs[i])[1], dim)
            vae.fit(xs[i], batch_size=50, epochs=10, verbose=False)
            transformed = encoder.predict(xs[i], verbose=False)
            model = utils.create_adult_model(dim, 2) if i == 0 else utils.create_wine_model(dim, 5)
            start = time.time()
            model.fit(transformed[:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            vae_samples[i].append(time.time() - start)

    times = [
        np.mean(none_samples[0]),
        np.mean(none_samples[1]),
        np.mean(pca_samples[0]),
        np.mean(pca_samples[1]),
        np.mean(ica_samples[0]),
        np.mean(ica_samples[1]),
        0,
        np.mean(rp_samples[1]),
        np.mean(vae_samples[0]),
        np.mean(vae_samples[1])
    ]

    times_err = [
        np.std(none_samples[0]) / 2,
        np.std(none_samples[1]) / 2,
        np.std(pca_samples[0]) / 2,
        np.std(pca_samples[1]) / 2,
        np.std(ica_samples[0]) / 2,
        np.std(ica_samples[1]) / 2,
        0,
        np.std(rp_samples[1]) / 2,
        np.std(vae_samples[0]) / 2,
        np.std(vae_samples[1]) / 2
    ]

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Influence of feature transformation on the NN training time')
    color = []
    for _ in range(5):
        color.append('tab:blue')
        color.append('tab:orange')
    x = []
    count = 1
    for _ in range(5):
        x.append(count)
        count += 0.5
        x.append(count)
        count += 1
    plot.bar(x, times, color=color, width=0.75, yerr=times_err)
    x = []
    count = 1.25
    for _ in range(5):
        x.append(count)
        count += 1.5
    plot.xticks(x, ['None', 'PCA', 'ICA', 'RP', 'VAE'])
    plot.xlabel('Feature transformation method')
    plot.ylabel('Average training time (s)')
    plot.show()


def nn2(xs, ys, xs_test, ys_test, n_components, clf_constructor):
    ks = [0 for _ in range(10)]
    cataccs = [0 for _ in range(10)]

    ys = [to_categorical(ys[0]), to_categorical(ys[1])]
    ys_test = [to_categorical(ys_test[0]), to_categorical(ys_test[1])]

    for i in range(2):
        shape = np.shape(xs[i])[1]
        n_components[i] = shape
        model = utils.create_adult_model(shape, 2) if i == 0 else utils.create_wine_model(shape, 5)
        model.fit(xs[i][:10000], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
        cataccs[i] = model.evaluate(xs_test[i], ys_test[i], verbose=False)[1] * 100

    for k in range(2, 11):
        try:
            clf = clf_constructor(n_clusters=k)
        except:
            clf = clf_constructor(n_components=k)
        for i in range(2):
            pca = PCA(n_components=n_components[2 + i])
            transformed = pca.fit_transform(xs[i])
            transformed_test = pca.transform(xs_test[i])
            predict = to_categorical(clf.fit_predict(transformed[:10000]))
            predict_test = to_categorical(clf.predict(transformed_test[:10000]))
            input_dims = [n_components[2 + i], k]
            model = utils.create_mi_adult_model(input_dims, 2) if i == 0 else utils.create_mi_wine_model(input_dims, 5)
            model.fit([transformed[:10000], predict], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            catacc = model.evaluate([transformed_test, predict_test], ys_test[i], verbose=False)[1] * 100
            if catacc > cataccs[2 + i]:
                ks[2 + i] = k
                cataccs[2 + i] = catacc

            ica = FastICA(n_components=n_components[4 + i])
            transformed = ica.fit_transform(xs[i])
            transformed_test = ica.transform(xs_test[i])
            predict = to_categorical(clf.fit_predict(transformed[:10000]))
            predict_test = to_categorical(clf.predict(transformed_test[:10000]))
            input_dims = [n_components[4 + i], k]
            model = utils.create_mi_adult_model(input_dims, 2) if i == 0 else utils.create_mi_wine_model(input_dims, 5)
            model.fit([transformed[:10000], predict], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            catacc = model.evaluate([transformed_test, predict_test], ys_test[i], verbose=False)[1] * 100
            if catacc > cataccs[4 + i]:
                ks[4 + i] = k
                cataccs[4 + i] = catacc

            if i == 1:
                rp = GaussianRandomProjection(eps=0.95)
                transformed = rp.fit_transform(xs[i])
                transformed_test = rp.transform(xs_test[i])
                predict = to_categorical(clf.fit_predict(transformed[:10000]))
                predict_test = to_categorical(clf.predict(transformed_test[:10000]))
                input_dims = [np.shape(transformed)[1], k]
                model = utils.create_mi_wine_model(input_dims, 5)
                model.fit([transformed[:10000], predict], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
                catacc = model.evaluate([transformed_test, predict_test], ys_test[i], verbose=False)[1] * 100
                if catacc > cataccs[6 + i]:
                    ks[6 + i] = k
                    cataccs[6 + i] = catacc

            encoder, vae = utils.create_vae(np.shape(xs[i])[1], n_components[8 + i])
            vae.fit(xs[i], batch_size=50, epochs=10, verbose=False)
            transformed = encoder.predict(xs[i], verbose=False)
            transformed_test = encoder.predict(xs_test[i], verbose=False)
            predict = to_categorical(clf.fit_predict(transformed[:10000]))
            predict_test = to_categorical(clf.predict(transformed_test[:10000]))
            input_dims = [n_components[8 + i], k]
            model = utils.create_mi_adult_model(input_dims, 2) if i == 0 else utils.create_mi_wine_model(input_dims, 5)
            model.fit([transformed[:10000], predict], ys[i][:10000], batch_size=50, epochs=10, verbose=False)
            catacc = model.evaluate([transformed_test, predict_test], ys_test[i], verbose=False)[1] * 100
            if catacc > cataccs[8 + i]:
                ks[8 + i] = k
                cataccs[8 + i] = catacc

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Influence of feature transformation on the NN accuracy')
    color = []
    for _ in range(5):
        color.append('tab:blue')
        color.append('tab:orange')
    x = []
    count = 1
    for _ in range(5):
        x.append(count)
        count += 0.5
        x.append(count)
        count += 1
    plot.bar(x, cataccs, color=color, width=0.75)
    x = []
    count = 1.25
    for _ in range(5):
        x.append(count)
        count += 1.5
    plot.xticks(x, ['None', 'PCA', 'ICA', 'RP', 'VAE'])
    plot.xlabel('Feature transformation method')
    plot.ylabel('Categorical accuracy (%)')
    plot.show()


x_adult, y_adult, x_adult_test, y_adult_test = utils.import_adult()
x_wine, y_wine, x_wine_test, y_wine_test = utils.import_wine()

# K-MEANS

silhouette('adult', range(2, 15), x_adult)  # 1025s
cluster_breakdown('adult_kmeans', x_adult, KMeans(n_clusters=5, random_state=0).fit_predict(x_adult))  # 10s
cataccs('adult', range(2, 15), 2, x_adult, y_adult) # 91s

silhouette('wine', range(2, 15), x_wine)  # 2154s
cluster_breakdown('wine_kmeans', x_wine, KMeans(n_clusters=3, random_state=0).fit_predict(x_wine))  # 24s
cataccs('wine reviews', range(2, 15), 5, x_wine, y_wine)  # 577s

# EXPECTATION MAXIMIZATION

expectation_maximization_silhouettes('adult', x_adult)  # 1214s
cluster_breakdown('adult_em', x_adult, GaussianMixture(n_components=6).fit_predict(x_adult))  # 74s

expectation_maximization_silhouettes('wine', x_wine)  # 4719s
cluster_breakdown('wine_em', x_wine, GaussianMixture(n_components=3).fit_predict(x_wine))  # 74s

# PCA

pca('adult', x_adult, y_adult)  # 1s
pca_benchmark('adult', x_adult, KMeans(n_clusters=5))  # 535s
pca_benchmark('adult', x_adult, GaussianMixture(n_components=6))  # 519s

pca('wine', x_wine, y_wine)  # 4s
pca_benchmark('wine', x_wine, KMeans(n_clusters=3))  # 612s
pca_benchmark('wine', x_wine, GaussianMixture(n_components=3))  # 320s
mca('Wine reviews', x_wine, y_wine) # 62s
mca_benchmark('Wine reviews', x_wine, KMeans(n_clusters=3))  # 511s
mca_benchmark('Wine reviews', x_wine, GaussianMixture(n_components=3))  # 505s
pca_eigenvalues(x_adult, x_wine)  # 31s

# ICA

ica('Adult', x_adult, y_adult)  # 2s
ica_benchmark('adult', x_adult, KMeans(n_clusters=5))  # 1326s
ica_benchmark('adult', x_adult, GaussianMixture(n_components=6))  # 1340s
kurtosis('Adult', x_adult, 2)  # 2s
kurtosis('Adult', x_adult, 3)  # 3s
ica_projection('Adult', x_adult, 2)  # 3s
ica_projection('Adult', x_adult, 3)  # 2s

ica('Wine', x_wine, y_wine)  # 18s
ica_benchmark('wine', x_wine, KMeans(n_clusters=3))  # 1636s
ica_benchmark('wine', x_wine, GaussianMixture(n_components=3))  # 1700s
kurtosis('Wine reviews', x_wine, 2)  # 41s
kurtosis('Wine reviews', x_wine, 3)  # 43s
ica_projection('Wine', x_wine, 2)  # 39s
ica_projection('Wine', x_wine, 3)  # 41s

# rp

jlmd_search([100, 1000], ['Adult', 'Wine reviews'])  # 0s
rp('Wine reviews', x_wine, y_wine)  # 10S
rp_benchmark('Wine reviews', x_wine, KMeans(n_clusters=3))  # 1550s
rp_benchmark('Wine reviews', x_wine, GaussianMixture(n_components=3))  # 1700s

# VAE

vae('Adult', x_adult, y_adult)  # 38s
vae_benchmark('Adult', x_adult, KMeans(n_clusters=5))  # 951s
vae_benchmark('Adult', x_adult, GaussianMixture(n_components=6))  # 458s

vae('Wine reviews', x_wine, y_wine)  # 158s
vae_benchmark('Wine reviews', x_wine, KMeans(n_clusters=3))  # 554s
vae_benchmark('Wine reviews', x_wine, GaussianMixture(n_components=3))  # 1604s

# DIMENSIONALITY REDUCTION + CLUSTERING

reduction_clustering(
    [x_adult, x_wine],
    [y_adult, y_wine],
    [2, 5],
    [KMeans(n_clusters=5), KMeans(n_clusters=3)],
    [PCA(n_components=2), PCA(n_components=2)],
    [FastICA(n_components=2), FastICA(n_components=2)],
    [None, GaussianRandomProjection(eps=0.95)],
    [utils.create_vae(np.shape(x_adult)[1]), utils.create_vae(np.shape(x_wine)[1])]
)  # 202s
reduction_clustering(
    [x_adult, x_wine],
    [y_adult, y_wine],
    [2, 5],
    [GaussianMixture(n_components=6), GaussianMixture(n_components=3)],
    [PCA(n_components=3), PCA(n_components=2)],
    [FastICA(n_components=3), FastICA(n_components=3)],
    [None, GaussianRandomProjection(eps=0.95)],
    [utils.create_vae(np.shape(x_adult)[1]), utils.create_vae(np.shape(x_wine)[1])]
)  # 291s

# DIMENSIONALITY REDUCTION + NN

nn([x_adult, x_wine], [y_adult, y_wine], [x_adult_test, x_wine_test], [y_adult_test, y_wine_test])  # 2181s
nn_benchmark(
    [x_adult, x_wine],
    [y_adult, y_wine],
    [104, 1075, 8, 2, 9, 5, 0, 5, 2, 10]
)  # 1661s

# DIMENSIONALITY REDUCTION + CLUSTERING + NN

nn2(
    [x_adult, x_wine],
    [y_adult, y_wine],
    [x_adult_test, x_wine_test],
    [y_adult_test, y_wine_test],
    [104, 1075, 8, 2, 9, 5, 0, 5, 2, 10],
    KMeans
)  # 2372s
nn2(
    [x_adult, x_wine],
    [y_adult, y_wine],
    [x_adult_test, x_wine_test],
    [y_adult_test, y_wine_test],
    [104, 1075, 8, 2, 9, 5, 0, 5, 2, 10],
    GaussianMixture
)  # 2179s
