# REPTA EPILEPSIA AMB ELECTROENCEFALOGRAMA


## INTRODUCCIÓ

Analitzarem diferents senyals cerebrals, per tal d'entrenar un model d'intel·ligència artificial que pugui detectar si, una persona està patint o no un atac d'epilèpsia, per tal de poder classificar quan una persona rep o no un atac s'utilitzaran finestres.
Les finestres són un grup de mostres de senyals en el temps d'un pacient en concret, i estan etiquetades amb "no té epilèpsia" i "té epilèpsia". Aquestes finestres etiquetades s'hauran de tractar perquè l'espai de temps que representa es pugui tractar com un únic senyal i no com un conjunt de senyals i així poder entrenar el model.
L'ordre del senyal té importàncies, ja que un conjunt de finestres d'un mateix pacient són contínues en el mateix temps. La nostra xarxa ha d'aprendre del passat, pel fet que per una mateixa finestra té diferents senyals que tenen un ordre temporal. Un senyal del passat influeix en els senyals següents. 

Primer hem decidit usar encoders per fer la classificació de finestres, ja que és més simple i podem veure si la part de tractament d'imatge i creació de finestres és correcte i a més a més, comprovem que la IA pugui entrenar del nostre model. Un cop realitzat aquest procés, passarem l'output, com a input a la xarxa LSTM i ens quedarem amb el model que millors resultats tingui. 

L'encoder canviarà l'embedding de característiques del senyal rebut per poder extreure les característiques d'una manera en la qual siguin més fàcils de classificar i poder diferenciar entre les diferents classes que rep el model.

En canvi, la LSTM és un tipus de RNN, amb la capacitat recordar patrons a llarg termini, ja que les RNN tradicionals sempre tenen problemes amb el gradient amb seqüències llargues. LSTM té l’habilitat de posseir cel·les amb memòria, és a dir, cel·les que estan regulades per cel·les que controlen la informació que surt i entra, fent així que la xarxa pugui oblidar informació de poca importància i mantenint tota aquella informació que aporta i permet aprendre i millorar el model.


## OBJECTIUS


L'objectiu d'aquest treball és captar si una persona està tenint un atac o no d'epilèpsia i poder classificar-lo, basant-nos en els senyals cerebrals.

També tenim com a objectiu tractar el gran desequilibri de les dades, ja que a la base de dades hi ha una gran quantitat de finestres sense presència d'epilèpsia i poques amb presència. S'ha de tractar el dataset per equilibrar les dades o fer que aquest desequilibri no afecti l'aprenentatge del model i que no quedi esbiaixat.

Per altra banda, tenim com a objectiu definir bé quines són les mètriques que volem usar, i com s'avaluarà el model, per tal que, els resultats tinguin pes per prendre decisions.
Complir amb aquests objectius en l'àmbit mèdic és molt important, ja que hi ha vides en joc i la presa d'una mala decisió per una errònia interpretació del model pot ser molt costós pels pacients.


## BASE DE DADES 
La base de dades consisteix en 24 pacients on per cadascun hi ha diverses hores de senyals EEG, les sigles EEG corresponen a "Electroencefalograma". L'EEG és una tècnica de neuroimatge que mesura l'activitat elèctrica del cervell.

Normalment, els senyals es registren utilitzant elèctrodes col·locats al cuir cabellut. Cada pacient té el format d'emmagatzematge Parquet, que és un format de fitxer d'emmagatzematge columnar altament optimitzat per usar-se amb marcs de processament de dades massives. Es fa servir habitualment en el context de l'emmagatzematge i el processament eficient de conjunts de dades grans.

Aquesta base de dades ha estat tractada de la següent manera:
Hem fet un subsampling per pacient; per cada pacient hem agafat la mateixa quantitat de dades per les diferents classes i així poder fer un entrenament amb les dades equilibrades.


## DISTRIBUCIÓ

El GitHub l'hem distribuït de la següent manera:

    (cunado acabemos todos los cambios de dsitribucon i carpetas lo ponemos)

## PROCEDIMENT

### XARXA NEURONAL: ENCODER I CLASSIFICACIÓ

Ens hem basat en la següent estructura per fer la xarxa neuronal.
![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/4570c698-fa07-48bb-ae09-f00959730f8b)

La raó d'aquesta elecció ha estat perquè així podiem reutilitzar el codi de l'anterior repte, així doncs l'únic canvi significatiu que hem fet ha estat afegir un data fusion unit a la sortida de l'encoder i fer que l'entrada de l'encoder sigui un canal en comptes dels 16 a la vegada.  



L'encoder és responsable de transformar la entrada en una representació que pugui ser utilitzada per la resta del model per realitzar tasques especifiques, en el nostre cas és la classificació de si una persona està rebent o no un atac d'epilepsia. En el nostre cas, l'encoder està composat per capes de convolució, seguides de funcions d'activació ReLU i capes de max pooling. Cada capa de convolució aprèn a extreure les carecteristiques necessaries per poder identificar les persones que tenen un atac. Després de que la secuencia del senyal pasin per encoder crea un embedding amb carecteritiques de la secuencia espacial del senyal del cervell. Que permet fer una clasificació. Optem per utilitzar capes convolucionals 2D per la seva eficàcia en la reducció de dimensionalitat i en la captura de diversos patrons presents al senyal. Per assegurar-nos d'un encoder precís i robust, decidim limitar el nombre de filtres utilitzats. Tant el padding com l'stride es configuren a 1 per mantenir la dimensionalitat (padding) i evitar ometre cap píxel (stride). Tant el padding com l'stride es configuren a 1 per mantenir la dimensionalitat (padding) i evitar ometre cap píxel (stride).




### explicar el data loader --> por finestra adan: no entien bien que es esto

El dataloader que hem utilitzat per entrenar el nostre model d'autoencoders és un conjunt de finestres, representen 1 segon de visualització del senyal del cervell. El model transforma aquesta finestra en un embedding on extreu les carecteristiques en un tipus de dades. S'han agafat de manera blanzajada entre les dues classes per evitar que el nostre model de classifiació quedi esbiaxat cap a una classe.




La xarxa conte la següent estructura:

    - Convolucional - 2D(1, 32, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool - 2D: ((1,2), stride=(1,2), padding=(0,1))
    - Convolucional - 2D (32, 64, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool - 2D ((1,2), stride=(1,2), padding=(0,1))
    - Convolucional - 2D (64, 128, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool - 2D ((1,2), stride=(1,2), padding=(0,1))
    - Convolucional - 2D (128, 256, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool ((1,2), stride=(1,2), padding=(0,1))


Les capes per fer la fusió de cracaterístiques tenen la següent estructura:

    Convolucional - 2D (21,1,(1,1)
    ReLu
    AdaptiveAvgPool2d ((256,4))
    Flatten()

La idea és fusionar els 21 canals en 1 de sol amb la capa de convolució, seguida d'una funció d'activació ReLU per tal de no aplicar linealitat. A continuació s'utilitza un adaptive average pooling, per tal d'ajustar automàticamnet la dimensió de la sortida i per últim apliquem un flatten, per tal d'aplanar la sortida a un vector unidimensional.

#### Paràmetres

    Els paràmetres utilitzats per tal d'arribra a un resultat robust i coherent han estat els següents:
        
    Èpoques: 300
    Optmimizer: Adamax
    lr: 0'001
    Criterion: BCEWithLogitsLoss


- BCEWithLogitsLoss: Aquesta pèrdua combina una capa sigmoide i la BCELoss en una sola classe. Aquesta versió és més estable numèricament que utilitzar un sigmoide senzill seguit d'un BCELoss ja que, combinant les operacions en una sola capa, aprofitem el truc log-sum-exp per a l'estabilitat numèrica. El BCELoss és una mesura que l'entropia creuada binaria entre les probabilitats d'entrada i de la sortida.


### poner las otras versiones por paciente 
### LSTM

Una LSTM com a classificador té la capacitat de procesar seqüencies, aprendre dependències a llarg termini i generar una representació interna de les dades que permet realitzar tasques de clasificació on hi ha present patrons seqüencials en les dades d'entrada. Això fa que sigui de gran utilitat en el nostre cas on analitzem conjunt de dades que tracten sobre la visualització durant hores de senyals del servei on l'impuls actual depent de l'impuls anterior. Si hi ha alguna seqüència o patró alhora d'obtenir l'atac el detectarem amb LSTM. 

La xarxa conte la següent estructura:
    pner estructura cambiada si no se pone la del encoder

    - Convolucional - 2D(1, 32, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool - 2D: ((1,2), stride=(1,2), padding=(0,1))
    - Convolucional - 2D (32, 64, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool - 2D ((1,2), stride=(1,2), padding=(0,1))
    - Convolucional - 2D (64, 128, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool - 2D ((1,2), stride=(1,2), padding=(0,1))
    - Convolucional - 2D (128, 256, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool ((1,2), stride=(1,2), padding=(0,1))

## RESULTATS


## CONCLUSIÓ

