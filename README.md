# REPTA EPILEPSIA AMB ELECTROENCEFALOGRAMA


## INTRODUCCIÓ

Analitzarem diferents senyals cerebrals, per tal d'entrenar un model d'intel·ligència artificial que pugui detectar si, una persona està patint o no un atac d'epilèpsia, per tal de poder classificar quan una persona rep o no un atac s'utilitzaran finestres.
Les finestres són un grup de mostres de senyals en el temps d'un pacient en concret, i estan etiquetades amb "no té epilèpsia" i "té epilèpsia". Aquestes finestres etiquetades s'hauran de tractar perquè l'espai de temps que representa es pugui tractar com un únic senyal i no com un conjunt de senyals i així poder entrenar el model.
L'ordre del senyal té importàncies, ja que un conjunt de finestres d'un mateix pacient són contínues en el mateix temps. La nostra xarxa ha d'aprendre del passat, pel fet que per una mateixa finestra té diferents senyals que tenen un ordre temporal. Un senyal del passat influeix en els senyals següents. 

Primer hem decidit usar encoders per fer la classificació de finestres, ja que és més simple i podem veure si la part de tractament d'imatge i creació de finestres és correcte i a més a més, comprovem que la IA pugui entrenar del nostre model. Un cop realitzat aquest procés, passarem l'output, com a input a la xarxa LSTM i ens quedarem amb el model que millors resultats tingui. 

L'encoder canviarà l'embedding de característiques del senyal rebut per poder extreure les característiques d'una manera en la qual siguin més fàcils de classificar i poder diferenciar entre les diferents classes que rep el model.

En canvi, la LSTM és un tipus de RNN, amb la capacitat recordar patrons a llarg termini, ja que les RNN tradicionals sempre tenen problemes amb el gradient amb seqüències llargues. LSTM té l’habilitat de posseir cel·les amb memòria, és a dir, cel·les que estan regulades per cel·les que controlen la informació que surt i entra, fent així que la xarxa pugui oblidar informació de poca importància i mantenint tota aquella informació que aporta i permet aprendre i millorar el model.


## OBJECTIUS

L'objectiu d'aquest treball és veure si podem veure si una persona està tenint un atac o no d'epilepsia, per tal de saber si tenim els recursos i la capcitat per determinar que una persona està patint un atac amb els senyals cerebrals.
També tenim com a objectiu tractar el gran desequilibri de les dades ja que per cada pacient hi han una gran quantitat de finestres on les persones no estan rebent un atac epilepsia i molt poques on si s'ha esta patint un atac. S'ha de tractar el dataset per equilibrar les dades o fer que aquest desequilibri no afecti al model.
Com hi ha diferetns maneres de definir que és  train o test és a dir que podem fer un train i test segons el pacients, finestres o  atacs d'epilepsia.Creant aixì com objectiu fer un model complert que defineixi millor quines són les metriques i com s'evaluarà el model perquè els resultats tinguin pes per prendre decisions. Complir amb aquests objectius en l'ambit médic és molt important ja que hi ha vides en joc i la presa d'una mala desició per una erronea interpretació del model pot ser molt costós pels pacients. Fent que hi hagi una base per aplicacions futeres médiques erronea perjudicant a principalment als pacients. 


## BASE DE DADES 
La base de dades consisteix de 24 pacients on per cada pacient hi ha varies hores de senyals EEG les sigles EEG corresponen a "Electroencefalograma". L'EEG és una tècnica de neuroimatge que mesura l'activitat elèctrica del cervell. Normalment, els senyals es registren utilitzant elèctrodes col·locats al cuir cabellut. On hi ha moments on la persona està patint un atac d'epilepsia. Cada pacient té el format d'emmagatzematge Parquet format que és un format de fitxer d'emmagatzematge columnar altament optimitzat per utilitzar-se amb marcs de processament de dades massives. S'utilitza habitualment en el context de l'emmagatzematge i el processament eficient de conjunts de dades grans. A aquesta base de dades ha estat tractada de la següent manera hem fet un subsampling de per pacient on per cada pacient hem agafat la mateixa quantitat de dades per les diferetns clases y aixì poder fer un entrenament amb les dades equilibrades.

## DISTRIBUCIÓ

El GitHub l'hem distribuït de la següent manera:

    (cunado acabemos todos los cambios de dsitribucon i carpetas lo ponemos)

## PROCEDIMENT
### XARXA NEURONAL: ENCODER I CLASSIFICACIÓ

Aquesta part consisteix en la creació del encoder per poder fer la clasificació, abans de pasar les dades per les capes del encoder fusionarem els diferetns canals del senyal EEG que té cada finiestra perquè le model pogui aprendre. 

### LSTM


## RESULTATS


## CONCLUSIÓ





##
