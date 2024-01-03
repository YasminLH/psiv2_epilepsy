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

Ens hem basat en la següent estructura.
![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/4570c698-fa07-48bb-ae09-f00959730f8b)


L'encoder té la responsabilitat de transformar la imatge original en una versió de baixa dimensionalitat. En el nostre cas, l'encoder està composat per capes de convolució, seguides de funcions d'activació ReLU i capes de max pooling. Cada capa de convolució apren a identificar característiques específiques de l'entrada, mentre que les capes de max pooling redueixen progressivament les dimensions espacials, contribuint a crear una representació comprimida de la imatge original. Optem per utilitzar capes convolucionals 2D per la seva eficàcia en la reducció de dimensionalitat i en la captura de diversos patrons presents a les imatges. Per assegurar-nos d'un autoencoder desviat, decidim limitar el nombre de filtres utilitzats. Tant el padding com l'stride es configuren a 1 per mantenir la dimensionalitat (padding) i evitar ometre cap píxel (stride).

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


Les dades es fa amb la següents capes:

    Convolucional - 2D (21,1,(1,1)
    ReLu
    AdaptiveAvgPool2d ((256,4))
    Flatten()


####Paràmetres

    Els que he utlitzat en l'encoder han estat els següents:
        
    Èpoques: 300
    Optmimizer: Adamax
    lr: 0'001
    Criterion: BCEWithLogitsLoss




### LSTM


## RESULTATS


## CONCLUSIÓ





##
