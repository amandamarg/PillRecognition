import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils import zeroPadFront, get_terminology
from sklearn.cluster import KMeans
import matplotlib as mpl
from sklearn.decomposition import PCA
from torchvision.models import resnet50, ResNet50_Weights

def formatNDC9(label_code, product_code):
    return zeroPadFront(label_code, 5) + '-' + zeroPadFront(product_code, 4)

def normalize(images):
    mean = np.mean(images, axis=(0,1,2))
    std = np.std(images, axis=(0,1,2))
    return (images-mean)/std

def encodeImages(image_paths, model):
    #load and normalize images
    images = normalize(np.stack(image_paths.map(Image.open)))

    #convert to tensor and change dimensions to (n,c,w,h)
    images = torch.Tensor(images).permute(dims=(0,3,1,2))

    with torch.no_grad():
        encodings=model(images)
    
    return encodings


def plotClusters(x, labels, center, path="./cluster_plots"):
    n = len(labels)
    k = len(center)
    colors = list(mpl.colors.XKCD_COLORS.keys())[0:n]
    plt.scatter(x[:, 0], x[:, 1], c=list(map(lambda i: colors[i], labels)), s=10)
    plt.scatter(center[:, 0], center[:, 1], c='red', s=200, marker='X', label='Centroids')
    plt.title("k=" + str(k))
    plt.savefig(path + "/k=" + str(k) + ".png")
    plt.close()

class SPL_Terms:
    def __init__(self, path="./"):
        self.color = get_terminology(path + "/color.xml")
        self.color.set_index("code", inplace=True)
        self.shape = get_terminology(path + "/shape.xml")
        self.shape.set_index("code", inplace=True)

    def translateColorCode(self, spl_codes):
        if not spl_codes:
            return None
        elif isinstance(spl_codes, list):
            return [self.color.loc[c]["name"] if c in self.color.index else c for c in spl_codes]
        elif spl_codes in self.color.index:
            return self.color.loc[spl_codes]["name"]
        else:
            return spl_codes
    
    def translateShapeCode(self, spl_codes):
        if not spl_codes:
            return None
        elif isinstance(spl_codes, list):
            return [self.shape.loc[c]["name"] if c in self.shape.index else c for c in spl_codes]
        elif spl_codes in self.shape.index:
            return self.shape.loc[spl_codes]["name"]
        else:
            return spl_codes


def kmeans_elbow_plot(encodings, start_k, max_k=None, plot_clusters=False):
    if not max_k:
        max_k = start_k
    wcss=[]
    for k in range(start_k, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(encodings)
        center = kmeans.cluster_centers_
        wcss.append(kmeans.inertia_)
        if plot_clusters:
            plotClusters(encodings, labels, center)
    
    plt.plot(np.arange(start_k, max_k+1), wcss, marker="o")
    plt.title("Elbow Plot")
    plt.xlabel("k values")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.savefig("elbow_plot.png")
    plt.show()

def plotByShape(df, ax, spl_terms, title="Shape"):
    colors = list(mpl.colors.XKCD_COLORS.keys())
    for i, group in enumerate(df.groupby("SHAPE", dropna=False)):
        if pd.notna(group[0]):
            shape = spl_terms.translateShapeCode(group[0])
        else:
            shape = "N/A"
        data = np.stack(group[1]["2D_encoding"].values)
        ax.scatter(data[:,0], data[:,1], c=colors[i], label="shape:"+shape)
        ax.legend(bbox_to_anchor=(1.05, 1), fontsize=6)
        ax.set_title(title)

def plotByColor(df, ax, spl_terms, title="Color"):
    for group in df.groupby("COLOR", dropna=False):
        if pd.notna(group[0]):
            color = spl_terms.translateColorCode(group[0].split(";"))
            marker_style = {'marker':'o', 'markersize':10, 'color':color[0], 'markeredgecolor':'black'}
            if len(color) > 1:
                marker_style['fillstyle']='left'
                marker_style['markerfacecoloralt']=color[1]
        else:
            marker_style= {'marker':'.', 'markersize':10, 'color':'black'}
        data = np.stack(group[1]["2D_encoding"].values)
        ax.set_title(title)
        ax.plot(data[:,0], data[:,1], linestyle="None", **marker_style)

def plotIndividualClusters(df, k):
    spl_terms = SPL_Terms()
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(np.stack(df["2D_encoding"].values))
    center = kmeans.cluster_centers_
    df['center_num'] = labels
    os.makedirs("./k="+str(k), exist_ok=True)
    for i,group in enumerate(df.groupby('center_num')):
        center_coord = center[group[0]]
        fig, ax = plt.subplots(2)
        fig.tight_layout(pad=2)
        fig.set_size_inches(10,10)
        ax[0].scatter(center_coord[0], center_coord[1], color='red', s=100, marker='X', label='Centroids')
        ax[1].scatter(center_coord[0], center_coord[1], color='red', s=100, marker='X', label='Centroids')
        plotByColor(group[1], ax[0], spl_terms)
        plotByShape(group[1], ax[1], spl_terms)
        fig.suptitle("Cluster " + str(i))
        fig.savefig("./k=" + str(k) +"/cluster=" + str(i) + ".png")

def main(df, opt_k=None, overwrite=True):
    #initalize model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    #peel off last layer
    model.fc = torch.nn.Identity() 

    #encode images or load saved encodings
    dir_path = "./datasets/ePillId_data/classification_data/"
    if not overwrite and os.path.exists(dir_path + "encodings.pt"):
        with open(dir_path + "encodings.pt", "r") as file:
            encodings = torch.tensor(eval(file.read()))
        file.close()
    else:
        encodings = encodeImages(df["image_path"].map(lambda x: dir_path + x), model=model)
        torch.save(encodings, dir_path + "encodings.pt")

    #reduce encodings to 2 dimensions for plotting
    pca = PCA(2)
    df["2D_encoding"] = pca.fit_transform(encodings).tolist()

    kmeans_elbow_plot(np.stack(df["2D_encoding"].values), start_k=1, max_k=10, plot_clusters=True)

    if opt_k:
        plotIndividualClusters(df, opt_k)
    

if __name__ == "__main__":
    # get data
    all_labels = pd.read_csv("./datasets/ePillID_data/all_labels.csv")
    all_labels['NDC'] = all_labels.apply(lambda x: formatNDC9(x["label_code_id"], x["prod_code_id"]), axis=1)

    properties = pd.read_json("ePillId_properties.json")
    properties["NDC"] = properties["NDC"].apply(lambda x: formatNDC9(x.split("-")[0], x.split("-")[1]))
    properties.set_index("NDC", inplace=True)

    all_labels = all_labels.join(properties, on="NDC", how="left")
    segmented_nih_pills = all_labels[all_labels["image_path"].str.startswith("segmented_nih_pills_224")]

    sample = segmented_nih_pills.iloc[0:100]
    main(sample, opt_k=5)