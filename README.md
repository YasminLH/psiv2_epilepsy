# REPTE EPILÈPSIA AMB ELECTROENCEFALOGRAMA


## INTRODUCCIÓ

Analitzarem diferents senyals cerebrals, per tal d'entrenar un model d'intel·ligència artificial que pugui detectar si una persona està patint o no un atac d'epilèpsia. Per tal de poder classificar quan una persona rep o no un atac s'utilitzaran 4 nivells: les finestres com a nivell més baix, els intervals, recordings i pacients respectivament.

Les finestres són un grup de mostres de senyals en el temps d'un pacient en concret, i estan etiquetades amb "no té epilèpsia" i "té epilèpsia". També es troben etiquetades amb tres metadades més. A quin Interval temporal pertanyen, en quina gravació (o recording) pertanyen i de quin pacient son. Aquestes metadades són les que ens permetran fer aquests quatre nivells
Cada nivell és més genèric que l'anterior. És a dir el primer grup de Finestres que podem fer és Finestres que pertanyen al mateix Interval. Un Interval és un conjunt de Finestres que es troben en el mateix temps. Un recording és un grup de intervals (o un sol) que pertany a una gravació. Un pacient pot tenir vàries gravacions o recordings de dies diferents.

Procedimentalment, hem decidit tenir dos enfocs, el de l'encoder i l'LSTM. L'encoder canviarà l'embedding de característiques del senyal rebut per poder extreure les característiques d'una manera en la qual siguin més fàcils de classificar i poder diferenciar entre les diferents classes que rep el model.

En canvi, la LSTM és un tipus de RNN, amb la capacitat recordar patrons a llarg termini, ja que les RNN tradicionals sempre tenen problemes amb el gradient amb seqüències llargues. LSTM té l'habilitat de posseir cel·les amb memòria, és a dir, cel·les que estan regulades per cel·les que controlen la informació que surt i entra, fent així que la xarxa pugui oblidar informació de poca importància i mantenint tota aquella informació que aporta i permet aprendre i millorar el model.

Així doncs, primer de tot hem decidit usar encoders per fer la classificació de tots 4 nivells, ja que és més simple i com a primer apropament del problema, està bé, i pot ser molt efectiu a l'hora de la detecció de patrons complexos a les dades. Un cop realitzat aquest procés, farem el mateix pas, però amb una xarxa LSTM i en tots dos casos ens quedarem amb el model que generalitzi millor, ja que utilitzem el fenomen KFold.


## OBJECTIUS


L'objectiu d'aquest treball és captar si una persona està tenint un atac o no d'epilèpsia i poder classificar-lo, basant-nos en els senyals cerebrals.
Un altre objectiu bastant important, és classificar amb els diferents nivells i comparar quin és el millor d'ells.

També tenim com a objectiu tractar el gran desequilibri de les dades, ja que a la base de dades hi ha una gran quantitat de finestres, recordings i intervals sense presència d'epilèpsia i poques amb presència. S'ha de tractar el dataset per equilibrar les dades o fer que aquest desequilibri no afecti l'aprenentatge del model i que no quedi esbiaixat.

Per altra banda, tenim com a objectiu definir bé quines són les mètriques que volem usar, i com s'avaluarà el model, per tal que, els resultats tinguin pes per prendre decisions.
Complir amb aquests objectius en l'àmbit mèdic és molt important, ja que hi ha vides en joc i la presa d'una mala decisió per una errònia interpretació del model pot ser molt costós pels pacients.


## BASE DE DADES 
La base de dades consisteix en 24 pacients on per cadascun hi ha diverses hores de senyals EEG, les sigles EEG corresponen a "Electroencefalograma". L'EEG és una tècnica de neuroimatge que mesura l'activitat elèctrica del cervell.

Normalment, els senyals es registren utilitzant elèctrodes col·locats al cuir cabellut. Cada pacient té el format d'emmagatzematge Parquet, que és un format de fitxer d'emmagatzematge columnar altament optimitzat per usar-se amb marcs de processament de dades massives. Es fa servir habitualment en el context de l'emmagatzematge i el processament eficient de conjunts de dades grans.

Aquesta base de dades es troba de la següent manera:
-24 arxius en format .npz, un per cada pacient, que contenen les dades de les pròpies senyals. Cada arxiu és per tant un array amb format (numero de finestres, canals, temps). El nombre de finestres varia per cada pacient. Cada finestra té 21 canals i cada finestra està dividida en trossets de 128 temporalment. Cada finestra correspon a un segon de recording. Per tant, tenim les finestres dividides en 21 canals i 1/128 segons.
-24 arxius en fomat .parquet, un per cada pacient, que contenen les metadades de cada una de les finestres. Cada arxiu es tracta per tant d'un Dataframe on cada fila representa una finestra. Conte la classe, l'interval, el recording i el pacient al qual pertany cada finestra.
Els dos arxius es troben relacionat pel propi índex. És a dir, la primera entrada de l'array coincideix amb la primera entrada del Dataframe.

En total tenim 571905 finestres. Aquest és un nombre molt elevat de dades i si volguéssim utilitzar-les totes necessitaríem 200 GB de memòria ram aproximadament.
Aquestes finestres no es troben balancejades en cap de les formes. És a dir:
-Hi ha més dades negatives, sense epilèpsia, que positives. Un 84% de les dades són negatives.
-Els pacients no tenen tots el mateix nombre de recordings.
-Els recordings no tenen tots el mateix nombre d'intervals.
-Els intervals no tenen tots el mateix nombre de finestres.
-La distribució de les classes és diferent per cada una de les divisions. En un interval podem trobar un 90% de dades negatives i en un altre un 70%.

To això fa que es necessiti fer un tractament de les dades important abans que entrin al model. Parlarem en més detall *(nose)* en un altre apartat però bàsicament per evitar problemes:
Hem fet un subsampling per pacient; per cada pacient hem agafat la mateixa quantitat de dades per les diferents classes i així poder fer un entrenament amb les dades equilibrades.


## DISTRIBUCIÓ

El GitHub l'hem distribuït de la següent manera:
- codi.py: codi en python que conté el codi d'aquesta pràctica, tant el de LSTM com el de l'Encoder. 
- Losses: directori que conté les gràfiques de les losses
- Pickle.py: fitxter python que conté el codi on es generen les gràfiques de loss del train i test.
- Pickles: directori que conté els objectes pickle, per cadascun del models que tenim i per cadascun del nivells


## PROCEDIMENT

    
### DATALOADER:


----- EXPLIACION ADAN----------------------
Per carregar les nostres dades utlitzem un document que podem carregar en un dataframe on estan continguts totes les metadades del difernts pacients i arxius que ens diuen quina és les finestres que conten un atac epilepsia i quines no. Amb aquest arxiu fem el dataloader balnçajat agant la mateixa quantitat de finestres per les dues classes. Per tan aqui ja dividim entre train i test les diferents metades de manera balancejada fent aixì que ara amb el aquest contingut podem carregar les dades directament en el nostre model depent de on estan guardat les diferetns metades. Aixì no necesstiem carregar les dades que tenen un magnitud molt gran sino que amb les metadades al entrenar ja podem carragar-les directament en el model per ser entreant. Aquestes finetre estan guardades en parquets que es carregaran per er entreants en el model. 
Seguint el que hem dit anteriorment del dataloader hem fet dos dataloaders que un consisteix en separar per finestres sense agruparles per pacient per entreanr el nostre model és a dir que a qui pertany les finestres no té importancia. Per cada classe hi ha 2000 finestres. En canvi l'altre dataloader entrena per pacient per tan per cada pacient agrupem les seves finestres correponents on hi han atacs o no d'epilepsia i fem l'entrenament tenint en compte aquesta agrupació per entreanar el nostre model. Per cada pacient s'afageix la mateixa quantitat de finestres que siguin atac i que no ho siguin per aixì sigui un dataloader balancejat. De cada classe hi ha 450 finestres. 

------ DANIEL YA LO EPXLCIARÁS TU MEJR CON LA SEPARACION WINSOW, PACIENT RECORIDING, INTERVAL--------


### XARXA NEURONAL: ENCODER I LSTM

Ens hem basat en la següent estructura per fer la xarxa neuronal.
![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/4570c698-fa07-48bb-ae09-f00959730f8b)

La raó d'aquesta elecció ha estat perquè així podiem reutilitzar part del codi de l'anterior repte i perquè en comparació amb l'altre opció aquesta se'ns feia més fàcil d'implmentar.

La nostra xarxa neuronal conté 3 principals parts: encoder, fusion i fully connected.

#### ENCODER
L'encoder és responsable de transformar la entrada en una representació que pugui ser utilitzada per la resta del model per realitzar tasques especifiques, en el nostre cas és la classificació de si una persona està rebent o no un atac d'epilepsia
Conté varies capes convolucionals 2d, una de les raons per les quals em volgut utilitzar l'anterior estructura, ja que així podiem reutilitzar les mateixes capes 2d.
No obstant aixó, hem hagut de canviar l'entrada de l'encoder, ja que tal i com hem vist a l'estructura anterior, volem tractar els canals individualment i ja a posteriori fusionar-los. 
Cada capa de convolució aprèn a extreure les carecteristiques locals necessaries per poder identificar de les windows, pacients, intervals o dels recorings la presència d'un atac o no.
Seguit de les capes 2d, tenim les capes de max pooling, aquestes són molt utilitzades, ja que s'usen per reduïr la dimensió, mantenint les característiques importants. 
Tant el padding com l'stride es configuren a 1 per mantenir la dimensionalitat (padding) i evitar ometre cap píxel (stride). 
Aquesta combinació de la cpa convolucional i la max pooling, és la més acertada en aquests àmbits, ja que dona resultats molt específics i bons.

Per acabar tenim les funcions d'activació ReLU, que ajuden a al xarxa a aprendre representacions no lineals de les dades.

Un cop la seqüència del senyal passa per l'encoder crea un embedding amb característiques de la seqüència espaial del senyal del cervell, que ha estat reduït gradualment les dimensions de l'entrada i augmentat el número de canals i això permetrà una futura classificació.

#### FUSION
La idea és fusionar els 21 canals en 1 de sol amb la capa de convolució, seguida d'una funció d'activació ReLU per tal de no aplicar linealitat. A continuació s'utilitza un adaptive average pooling, per tal d'ajustar automàticamnet la dimensió de la sortida i per últim apliquem un flatten, per tal d'aplanar la sortida a un vector unidimensional.

#### FULLY CONNECTED
Aquesta és la capa fully connected, amb la que es realitzarà la classificació final. Conté dues capes totalment connectades, amb funcions d'activcaió ReLU entre elles. També inclou una capa dropout de 0.5, per la regularització i evitar l'overfitting, el que fa és apagar la meitat de les neurones durant el train de la xarxa. 
L'última capa conté dues neurones de sortida, ja que es tracta d'un problema de classificació binària (si epilèpsi o no epilèpsia)
Capa Fully Connected (fc):


La xarxa neuronal amb les 3 components té la següent estructura 
Encoder:

    - Convolucional - 2D(1, 128, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool - 2D: ((1,2), stride=(1,2), padding=(0,1))
    - Convolucional - 2D (128, 256, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool - 2D ((1,2), stride=(1,2), padding=(0,1))
    - Convolucional - 2D (256, 512, (1,3), stride=1, padding=(0,1))
    - ReLu
    - MaxPool - 2D ((1,2), stride=(1,2), padding=(0,1))
    
Fusion:

    Convolucional - 2D (21,1,(1,1)
    ReLu
    AdaptiveAvgPool2d ((256,4))
    Dropout: 0.5
    Flatten()

Fully connected

    Linear - (512 * 4, 256),
    ReLU
    Dropout - (0.5),  
    Linear - (256, 2)



#### Paràmetres

Els paràmetres utilitzats per tal d'arribar a un resultat robust i coherent han estat els següents:
        
    Èpoques: depèn del nivell i divisió, "s'explica més endavant"
    Optmimizer: Adamax
    lr: 0'001
    Criterion: BCEWithLogitsLoss
    dropout = 0.5

BCEWithLogitsLoss: Aquesta pèrdua combina una capa sigmoide i la BCELoss en una sola classe. Aquesta versió és més estable numèricament que utilitzar un sigmoide senzill seguit d'un BCELoss ja que, combinant les operacions en una sola capa, aprofitem el truc log-sum-exp per a l'estabilitat numèrica. El BCELoss és una mesura que l'entropia creuada binaria entre les probabilitats d'entrada i de la sortida.



### LSTM

Una LSTM com a classificador té la capacitat de procesar seqüencies, aprendre dependències a llarg termini i generar una representació interna de les dades que permet realitzar tasques de clasificació on hi ha present patrons seqüencials en les dades d'entrada. Això fa que sigui de gran utilitat en el nostre cas on analitzem conjunt de dades que tracten sobre la visualització durant hores de senyals del servei on l'impuls actual depent de l'impuls anterior. Si hi ha alguna seqüència o patró alhora d'obtenir l'atac el detectarem amb LSTM. 


La xarxa LSTM conte la següent estructura:

    Input_dim = 21
    Hidden_dim =  20
    Output_dim = 2
    Num_layers = 2
    Batch_size = 3
    Droput = 0.5
    Batch Normalization 
    Capa Linear

La raó de l'utilització del Dropout i el Batch normalization, és per prevenir l'overfitting i millorar la generalització del model.
El Dropout a 0.5, el que fa és inhibir a la meitat de les neurones a l'hora de l'entrenament i el batch normalization, s'encarrega d'estabilitzar i accelerar l'entrenament, normalitzant  les activacions a cada capa.

La mètrica per evaluar el rendiment del model, és la precisió, ja que és la mètrica òptima i correcte en aquest context mèdic de classificació de pacients per epilèpsia. 


## ÈPOQUES:
Per cada cas, tenim un número òptim d'èpoques, ja que per cada nivell i/o divisió cal una precisió o una altra. 

    - encoder window --> 30 epocas 
    - encoder interval -->10 epocas 
    - encoder pacient --> 14 epocas 
    - encoder recording --> 6 epocas
    
    - lstm  window --> 15 epocas 
    - lstm  interval -->10 epocas  
    - lstm  pacient --> 15 epocas  
    - lstm  recording --> 6 epocas
    
Per anar canviant de nivell o divisió, comentem o descomentem les línies que ens interessen, per exemple, vull executar el model amb "tipus" LSTM i "divisió" recording.

![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/8054e2fa-9e75-4862-8694-18a23ff47319)


## LOSS
Utilizem la BCEWithLogitsLoss, per calcular les discrepàncies entre les prediccions del model i les etiquetes reals i  obtenim la loss, específicament una loss per les dades train i una loss per les dades test. Això ho guardem en un objecte pickle, per tal de poder manipular més endavant com vulguem. El codi on generem els gràfics pertinents està al fitxer "pickle.py". Aquestes gràfiques ens serveixen per monitorar i veure quins dels models que hem generat ens va millor, generalitza més i evitar l'overfitting. 

## RESULTATS

### Resultats Loss
Utilizem les gràfiques de loss de train i test, per triar el millor model dels que ha generat KFold i hem entrenat.      

#### Encoder 
![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/d8331bd9-d828-42e8-bcd8-b8dc6b2a5c12)
![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/ccfb0cc8-8eb9-4a50-bba7-9b21c7ddd5b0)

![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/c39149c5-110f-4ae6-84df-90d18428cc06)
![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/3c08595d-ef69-4d58-8152-61ab76789722)

Podem veure que entre tots 4 models, destaca clarament el model 0 com a més eficient en generalització i que resisteix l'overfitting. Aquest model aprèn de manera precisa, robusta, adequada i correcte capturant la complexitat de les dades, però sense desviar-se de la tendència òptima de la loss tant de train com de test.


#### LSTM 

![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/85b48c8d-20b1-45b8-82ba-5937fd9c7639)
![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/308d59f8-0ff3-44c7-9ad4-a22ba955a50e)

![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/d8e0a25d-d96e-4fa1-b322-6f72265e2e61)
![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/e3b0c825-d84d-4ca1-ba02-0cd46aa54415)

Podem veure que entre tots 4 models, destaca clarament el model 3 com a més eficient en generalització i que resisteix l'overfitting. Aquest model a diferència dels altres aprèn de manera adequada, però a causa de les poques dades de test que tenim sembla que no aprèn de manera precisa i es queda més o menys estable quan arriba a l'època 10, ja que no té res més que aprendre.     
Segueix la més o menys la tendència òptima de la loss tant de train com de test.

Com a resultat final podem veure comparant les losses que la xarxa neuronal Encoder, va millor que la del LSTM, això podria ser que l'LSTM no estigui explotant de manera òptima  les relacions temporals de les dades o que tingui una arquitectura molt més complexa i fa que li costi més aprendre de manera molt més precisa els patrons locals. 
Per exemple l'encoder té capes específiques que aprenen patrons locals i capes que disminueixen la dimensionalitat, per tal de eliminar la complexitat i deixar només aquelles característiques més importants.

** Les altres losses del nivells recording, interval i pacient estan a la carpeta losses. **

Els millors resultats són els que s'han vist anteriorment, és ha dir el que fa la divisió per window.

### Resultats Encoder millor Model 


### Resultats LSTM millor Model 




## CONCLUSIÓ



