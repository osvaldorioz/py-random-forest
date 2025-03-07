#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>

namespace py = pybind11;
using namespace Eigen;

class DecisionTree {
public:
    void fit(const MatrixXd& X, const VectorXd& y) {
        // Implementación simple de un árbol de decisión
        threshold = X.col(0).mean();
        left_mean = y(X.col(0).array() <= threshold).mean();
        right_mean = y(X.col(0).array() > threshold).mean();
    }

    VectorXd predict(const MatrixXd& X) const {
        VectorXd predictions(X.rows());
        for (int i = 0; i < X.rows(); ++i) {
            predictions(i) = (X(i, 0) <= threshold) ? left_mean : right_mean;
        }
        return predictions;
    }

private:
    double threshold;
    double left_mean;
    double right_mean;
};

class RandomForest {
public:
    RandomForest(int n_trees) : n_trees(n_trees) {}
    
    void fit(py::array_t<double> X_np, py::array_t<double> y_np) {
        auto X = X_np.unchecked<2>();
        auto y = y_np.unchecked<1>();
        
        int n_samples = X.shape(0);
        int n_features = X.shape(1);
        
        MatrixXd X_eigen(n_samples, n_features);
        VectorXd y_eigen(n_samples);
        
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                X_eigen(i, j) = X(i, j);
            }
            y_eigen(i) = y(i);
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (int i = 0; i < n_trees; ++i) {
            std::vector<int> indices(n_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            
            MatrixXd X_bootstrap(n_samples, n_features);
            VectorXd y_bootstrap(n_samples);
            
            for (int j = 0; j < n_samples; ++j) {
                X_bootstrap.row(j) = X_eigen.row(indices[j]);
                y_bootstrap(j) = y_eigen(indices[j]);
            }
            
            DecisionTree tree;
            tree.fit(X_bootstrap, y_bootstrap);
            trees.push_back(tree);
        }
    }
    
    py::array_t<double> predict(py::array_t<double> X_np) {
        auto X = X_np.unchecked<2>();
        int n_samples = X.shape(0);
        MatrixXd X_eigen(n_samples, X.shape(1));
        
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < X.shape(1); ++j) {
                X_eigen(i, j) = X(i, j);
            }
        }
        
        VectorXd predictions = VectorXd::Zero(n_samples);
        for (const auto& tree : trees) {
            predictions += tree.predict(X_eigen);
        }
        predictions /= trees.size();
        
        return py::array_t<double>(predictions.size(), predictions.data());
    }

private:
    int n_trees;
    std::vector<DecisionTree> trees;
};

PYBIND11_MODULE(rfe_module, m) {
    py::class_<RandomForest>(m, "RandomForest")
        .def(py::init<int>())
        .def("fit", &RandomForest::fit)
        .def("predict", &RandomForest::predict);
}
