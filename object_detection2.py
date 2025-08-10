import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from skimage.io import imread
from skimage.transform import resize, rescale
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
import skimage.color
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


# === 1. Prepare Dataset ===
def resize_all(src, pklname, include, width=80, height=None):
    height = height if height else width
    data = {
        'description': f'resized ({width}x{height}) animal images in rgb',
        'label': [],
        'filename': [],
        'data': []
    }

    pklname = f'{pklname}_{width}x{height}px.pkl'

    for subdir in os.listdir(src):
        if subdir in include:
            print('Processing:', subdir)
            current_path = os.path.join(src, subdir)
            for file in os.listdir(current_path):
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    img = imread(os.path.join(current_path, file))
                    img = resize(img, (width, height))
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(img)
    joblib.dump(data, pklname)



# === 2. Resize & Save Dataset ===
base_name = 'animal_faces'
width = 80
include = {'Bear', 'Cat', 'Cow', 'Dog', 'Donkey', 'Duck', 'Elephant', 'Fox', 'Goat', 'Lion', 'Tiger', 'Wolf'}
data_path = r'C:\Users\ASUS\Desktop\4-1research\images'

resize_all(src=data_path, pklname=base_name, width=width, include=include)



# === 3. Load Pickle Dataset ===
data = joblib.load(f'{base_name}_{width}x{width}px.pkl')

print('Number of samples:', len(data['data']))
print('Keys:', list(data.keys()))
print('Description:', data['description'])
print('Image shape:', data['data'][0].shape)
print('Labels:', np.unique(data['label']))
label_counts = Counter(data['label'])
print('Label counts:')
for label, count in label_counts.items():
    print(f'{label}: {count}')



# === 4. Plot sample image per label ===
labels = np.unique(data['label'])
fig, axes = plt.subplots(1, len(labels), figsize=(15, 4))
fig.tight_layout()

for ax, label in zip(axes, labels):
    idx = data['label'].index(label)
    ax.imshow(data['data'][idx])
    ax.axis('off')
    ax.set_title(label)
plt.show()



# === HOG visualization on sample image ===
dog_img = imread('C:/Users/ASUS/Desktop/4-1research/images/Dog/2a8a6a6050 - Copy.jpg', as_gray=True)
dog_img_rescaled = rescale(dog_img, 1/3, mode='reflect')

dog_hog_features, dog_hog_image = hog(
    dog_img_rescaled,
    pixels_per_cell=(14, 14),
    cells_per_block=(2, 2),
    orientations=9,
    visualize=True,
    block_norm='L2-Hys'
)

fig, ax = plt.subplots(1, 2, figsize=(8, 6))
for a in ax:
    a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

ax[0].imshow(dog_img_rescaled, cmap='gray')
ax[0].set_title('dog')
ax[1].imshow(dog_hog_image, cmap='gray')
ax[1].set_title('hog')
plt.show()

print('Number of pixels:', dog_img_rescaled.shape[0] * dog_img_rescaled.shape[1])
print('Number of hog features:', dog_hog_features.shape[0])



# === 5. Train/Test split ===
X = np.array(data['data'])
y = np.array(data['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)



# === 6. Relative distribution bar plot ===
def plot_bar(y, loc='left', relative=True):
    width = 0.35
    n = -0.5 if loc == 'left' else 0.5

    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]
    counts = counts[sorted_index]

    if relative:
        counts = 100 * counts / len(y)
        ylabel_text = '% count'
    else:
        ylabel_text = 'count'

    xtemp = np.arange(len(unique))
    plt.bar(xtemp + n * width, counts, align='center', alpha=0.7, width=width)
    plt.xticks(xtemp, unique, rotation=45)
    plt.xlabel('animal type')
    plt.ylabel(ylabel_text)

plt.suptitle('Relative amount of photos per type')
plot_bar(y_train, loc='left')
plot_bar(y_test, loc='right')
plt.legend([f'train ({len(y_train)} photos)', f'test ({len(y_test)} photos)'])
plt.show()



# === 7. Define Transformers ===
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([skimage.color.rgb2gray(img) for img in X])

class HogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(14, 14),
                 cells_per_block=(2, 2), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def local_hog(img):
            return hog(img,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
        features = [local_hog(img) for img in X]
        return np.array(features)



# === 8. Preprocess & Train ===
grayify = RGB2GrayTransformer()
hogify = HogTransformer(pixels_per_cell=(14, 14), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')
scalify = StandardScaler()

X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_prepared, y_train)



# === 9. Evaluate on Test Data ===
X_test_gray = grayify.transform(X_test)
X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)

y_pred = sgd_clf.predict(X_test_prepared)



# Accuracy output like in notebook cell [16]
print(np.array(y_pred == y_test)[:25])
print('')
accuracy = 100*np.sum(y_pred == y_test)/len(y_test)
print('Percentage correct: ', accuracy)



def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100 * cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()
    np.fill_diagonal(cmx_zero_diag, 0)

    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(12, 3)
    [a.set_xticks(range(len(cmx)+1)) for a in ax]
    [a.set_yticks(range(len(cmx)+1)) for a in ax]

    im1 = ax[0].imshow(cmx, vmax=vmax1)
    ax[0].set_title('as is')
    im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
    ax[1].set_title('%')
    im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
    ax[2].set_title('% and 0 diagonal')

    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) for divider in dividers]

    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.tight_layout()
    plt.show()


# === 10. Small yes/no confusion matrix example ===
labels_example = ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no',  'no', 'no', 'no']
predictions_example = ['yes', 'yes', 'yes', 'yes', 'no',  'no',  'yes', 'no', 'no', 'no']

df_example = pd.DataFrame(
    np.c_[labels_example, predictions_example], 
    columns=['true_label', 'prediction']
)
print('\nExample yes/no predictions:')
print(df_example)

label_names_example = ['yes', 'no']
cmx_example = confusion_matrix(labels_example, predictions_example, labels=label_names_example)
df_cmx_example = pd.DataFrame(cmx_example, columns=label_names_example, index=label_names_example)
df_cmx_example.columns.name = 'prediction'
df_cmx_example.index.name = 'label'
print('\nExample confusion matrix:')
print(df_cmx_example)

plt.imshow(cmx_example)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.title('Example Confusion Matrix (yes/no)')
plt.show()

# === 11. Animal classification confusion matrix ===
label_names = sorted(np.unique(y_test))
cmx = confusion_matrix(y_test, y_pred, labels=label_names)
df_cmx = pd.DataFrame(cmx, columns=label_names, index=label_names)
df_cmx.columns.name = 'prediction'
df_cmx.index.name = 'label'
print('\nAnimal classification confusion matrix:')
print(df_cmx)

# Plot simple confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=label_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.tight_layout()
plt.title('Animal Classification Confusion Matrix')
plt.show()

# Plot detailed comparison (3-panel) confusion matrix
plot_confusion_matrix(cmx)

# Print label order
print('\nLabel order:', sorted(np.unique(y_test)))



# === 12. Show prediction vs true in DataFrame ===
result_df = pd.DataFrame({
    'true_label': y_test,
    'prediction': y_pred
})
print(result_df.head(10))


from sklearn.pipeline import Pipeline
from sklearn import svm
 
HOG_pipeline = Pipeline([
    ('grayify', RGB2GrayTransformer()),
    ('hogify', HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys')
    ),
    ('scalify', StandardScaler()),
    ('classify', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
])
 
clf = HOG_pipeline.fit(X_train, y_train)
print('Percentage correct: ', 100*np.sum(clf.predict(X_test) == y_test)/len(y_test))

