# Dehazer

Bonjour ! Ceci est notre **projet Dehazer** (débrumage d’images).

##  Structure du projet

Le code principal se trouve dans :
**code_src/complet_dehazer/main.py**

Après avoir installé toutes les dépendances, il vous suffit d’exécuter :

_python main.py_


Cela lancera une **interface visuelle en terminal** qui vous permettra de :

- Choisir la méthode de débrumage à appliquer  
- Gérer et configurer le pipeline de traitement  
- Comparer deux images en calculant leur différence pour observer plus clairement les variations entre différentes méthodes  

## 🖼️ Base d’images

Nos principales images de test se trouvent dans :

**hazed_images/I_hazed_images**

**hazed_images/very_hazed_images**

Vous pouvez les utiliser pour tester le fonctionnement du projet !

Si vous voulez plus de ressources imageries pour tester, vous pouvez utiliser _curl <lien>_ pour télécharger les datasets dessous:

- I-Haze: Images brumées artificielles en intérieur **http://www.vision.ee.ethz.ch/ntire18/i-haze/I-HAZE.zip**
- O-Haze: Images brumées artificielles en extérieur **http://www.vision.ee.ethz.ch/ntire18/o-haze/O-HAZE.zip**
- NH-Haze: Images brumées de manière non homogène **https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip**
- D-Haze: Images brumées dépendant de la profondeur **http://ancuti.meo.etc.upt.ro/D_Hazzy_ICIP2016/D-HAZY_DATASET.zip**
