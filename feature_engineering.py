import numpy as np
import grs1915_utils as ut
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle


def transform_chi(labels):
    labels = np.array(labels)  # Ensure labels is a numpy array
    labels[np.isin(labels, ["chi1", "chi2", "chi3", "chi4"])] = "chi"
    return labels


def load_features(datadir, tseg, ranking=None):
    features_train_full = np.loadtxt(f"{datadir}grs1915_{tseg}s_features_train.txt")
    features_test_full = np.loadtxt(f"{datadir}grs1915_{tseg}s_features_test.txt")
    features_val_full = np.loadtxt(f"{datadir}grs1915_{tseg}s_features_val.txt")

    labels_train_full = np.array(ut.conversion(f"{datadir}grs1915_{tseg}s_labels_train.txt")[0])
    labels_test_full = np.array(ut.conversion(f"{datadir}grs1915_{tseg}s_labels_test.txt")[0])
    labels_val_full = np.array(ut.conversion(f"{datadir}grs1915_{tseg}s_labels_val.txt")[0])

    tstart_train_full = np.loadtxt(f"{datadir}grs1915_{tseg}s_tstart_train.txt")
    tstart_test_full = np.loadtxt(f"{datadir}grs1915_{tseg}s_tstart_test.txt")
    tstart_val_full = np.loadtxt(f"{datadir}grs1915_{tseg}s_tstart_val.txt")

    nseg_train_full = np.loadtxt(f"{datadir}grs1915_{tseg}s_nseg_train.txt")
    nseg_test_full = np.loadtxt(f"{datadir}grs1915_{tseg}s_nseg_test.txt")
    nseg_val_full = np.loadtxt(f"{datadir}grs1915_{tseg}s_nseg_val.txt")

    lc_all_full = ut.getpickle(f"{datadir}grs1915_{tseg}s_lc_all.dat")
    hr_all_full = ut.getpickle(f"{datadir}grs1915_{tseg}s_hr_all.dat")

    lc_train_full, lc_test_full, lc_val_full = lc_all_full["train"], lc_all_full["test"], lc_all_full["val"]
    hr_train_full, hr_test_full, hr_val_full = hr_all_full["train"], hr_all_full["test"], hr_all_full["val"]

    delete_indices = {
        "train": np.where(features_train_full[:, 17] >= 20)[0],
        "val": np.where(features_val_full[:, 17] >= 20)[0],
        "test": np.where(features_test_full[:, 17] >= 20)[0],
    }

    for key in delete_indices:
        if len(delete_indices[key]) > 0:
            print(f"Found outlier in {key} set.")
            exec(f"features_{key}_full = np.delete(features_{key}_full, delete_indices[key], axis=0)")
            exec(f"labels_{key}_full = np.delete(labels_{key}_full, delete_indices[key], axis=0)")
            exec(f"lc_{key}_full = np.delete(lc_{key}_full, delete_indices[key], axis=0)")
            exec(f"hr_{key}_full = np.delete(hr_{key}_full, delete_indices[key], axis=0)")
            exec(f"tstart_{key}_full = np.delete(tstart_{key}_full, delete_indices[key], axis=0)")
            exec(f"nseg_{key}_full = np.delete(nseg_{key}_full, delete_indices[key], axis=0)")

    labels_train_full = transform_chi(labels_train_full)
    labels_test_full = transform_chi(labels_test_full)
    labels_val_full = transform_chi(labels_val_full)

    return {
        "features": {"train": features_train_full, "test": features_test_full, "val": features_val_full},
        "labels": {"train": labels_train_full, "test": labels_test_full, "val": labels_val_full},
        "lc": {"train": lc_train_full, "test": lc_test_full, "val": lc_val_full},
        "hr": {"train": hr_train_full, "test": hr_test_full, "val": hr_val_full},
        "tstart": {"train": tstart_train_full, "val": tstart_val_full, "test": tstart_test_full},
        "nseg": {"train": nseg_train_full, "val": nseg_val_full, "test": nseg_test_full},
    }


def choose_label(label):
    chaotic = {"beta", "lambda", "kappa", "mu"}
    deterministic = {"theta", "rho", "alpha", "nu", "delta"}
    stochastic = {"phi", "gamma", "chi"}

    if label in chaotic:
        return "chaotic"
    elif label in deterministic:
        return "deterministic"
    elif label in stochastic:
        return "stochastic"
    else:
        return label


def convert_labels_to_physical(labels):
    return {
        "train": np.array([choose_label(l) for l in labels["train"]]),
        "val": np.array([choose_label(l) for l in labels["val"]]),
        "test": np.array([choose_label(l) for l in labels["test"]]),
    }


def scale_features(features):
    features_all = np.vstack([features["train"], features["val"], features["test"]])
    scaler = StandardScaler().fit(features_all)

    return {
        "train": scaler.transform(features["train"]),
        "test": scaler.transform(features["test"]),
        "val": scaler.transform(features["val"]),
    }


def greedy_search(datadir, seg_length_supervised=1024.0, n_folds=5):
    data = load_features(datadir, seg_length_supervised)
    features, labels = data["features"], data["labels"]

    features_lb, labels_lb = scale_features(features), labels
    features_train, features_val, features_test = features_lb["train"], features_lb["val"], features_lb["test"]
    labels_train, labels_val = labels_lb["train"], labels_lb["val"]

    score_all, feature_ranking = [], []
    nfeatures = list(range(features_train.shape[1]))

    for i in range(features_train.shape[1]):
        print(f"I am on the {i}th loop")
        best_score, best_feature = -1, None

        for j in nfeatures:
            ft, fv = features_train[:, [j]], features_val[:, [j]]
            fscaled_train = StandardScaler().fit_transform(ft)
            fscaled_val = StandardScaler().fit_transform(fv)

            lr = LogisticRegression(class_weight="balanced", multi_class="multinomial", solver="lbfgs")
            grid_lr = GridSearchCV(lr, {"C": [0.01, 0.1, 1.0, 10.0]}, cv=KFold(n_splits=n_folds), scoring="f1_weighted")
            grid_lr.fit(fscaled_train, labels_train)

            score = grid_lr.best_score_
            if score > best_score:
                best_score, best_feature = score, j

        feature_ranking.append(best_feature)
        nfeatures.remove(best_feature)
        score_all.append(best_score)

    with open(f"{datadir}grs1915_greedysearch_res.dat", "wb") as f:
        pickle.dump({"ranking": feature_ranking, "scores": score_all}, f)


if __name__ == "__main__":
    greedy_search("/scratch/daniela/data/grs1915/")
