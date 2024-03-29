# REPTE EPILÈPSIA AMB ELECTROENCEFALOGRAMA


## INTRODUCCIÓ

Analitzarem diferents senyals cerebrals, per tal d'entrenar un model d'intel·ligència artificial que pugui detectar si una persona està patint o no un atac d'epilèpsia. Per tal de poder classificar quan una persona rep o no un atac s'utilitzaran 4 nivells: les finestres com a nivell més baix, els intervals, recordings i pacients respectivament.

Les finestres són un grup de mostres de senyals en el temps d'un pacient en concret, i estan etiquetades amb "no té epilèpsia" i "té epilèpsia". També es troben etiquetades amb tres metadades més. A quin Interval temporal pertanyen, en quina gravació (o recording) pertanyen i de quin pacient son. Aquestes metadades són les que ens permetran fer aquests quatre nivells
Cada nivell és més genèric que l'anterior. És a dir el primer grup de Finestres que podem fer és Finestres que pertanyen al mateix Interval. Un Interval és un conjunt de Finestres que es troben en el mateix temps. Un recording és un grup d'intervals (o un sol) que pertany a una gravació. Un pacient pot tenir diverses gravacions o recordings de dies diferents.

Procedimentalment, hem decidit tenir dos enfocaments, el de l'encoder i l'LSTM. L'encoder canviarà l'embedding de característiques del senyal rebut per poder extreure les característiques d'una manera en la qual siguin més fàcils de classificar i poder diferenciar entre les diferents classes que rep el model.

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

- 24 arxius en format .npz, un per cada pacient, que contenen les dades de les pròpies senyals. Cada arxiu és per tant un array amb format (número de finestres, canals, temps). El nombre de finestres varia per cada pacient. Cada finestra té 21 canals i cada finestra està dividida en trossets de 128 temporalment. Cada finestra correspon a un segon de recording. Com a resultat, tenim les finestres dividides en 21 canals i 1/128 segons.

- 24 arxius en fomat. parquet, un per cada pacient, que contenen les metadades de cada una de les finestres. Cada arxiu es tracta, per tant, d'un Dataframe on cada fila representa una finestra. Conte la classe, l'interval, el recording i el pacient al qual pertany cada finestra.
Els dos arxius es troben relacionat pel mateix índex. És a dir, la primera entrada de l'array coincideix amb la primera entrada del Dataframe.

En total tenim 571905 finestres. Aquest és un nombre molt elevat de dades i si volguéssim utilitzar-les totes necessitaríem 200 GB de memòria ram aproximadament.
Aquestes finestres no es troben balancejades en cap de les formes. És a dir:

- Hi ha més dades negatives, sense epilèpsia, que positives. Un 84% de les dades són negatives.

- Els pacients no tenen tots el mateix nombre de recordings.

- Els pacients no tenen tots el mateix nombre d'intervals.

- Els intervals no tenen tots el mateix nombre de finestres.

- La distribució de les classes és diferent per cada una de les divisions. En un interval podem trobar un 90% de dades negatives i en un altre un 70%.

Tot això fa que es necessiti fer un tractament de les dades important abans que entrin al model. Parlarem en més detall posteriorment, però bàsicament per evitar problemes
hem fet un subsampling per pacient. Per cada pacient hem agafat la mateixa quantitat de dades per les diferents classes i així poder fer un entrenament amb les dades equilibrades.


## DISTRIBUCIÓ

El GitHub l'hem distribuït de la següent manera:
- codi.py: codi en python que conté el codi d'aquesta pràctica, tant el de LSTM com el de l'Encoder.
- Losses: directori que conté les gràfiques de les losses
- Pickle.py: fitxter python que conté el codi on es generen les gràfiques de loss del train i test.
- Pickles: directori que conté els objectes pickle, per cadascun dels models que tenim i per cadascun dels nivells


## PROCEDIMENT

    
### DATALOADING:

Com ja s'ha mencionat abans la càrrega de dades no és trivial, ja que no totes les dades caben a memòria ram. Per tant, la solució és fer un subsampling. La càrrega de dades és diferent per cada una de les 4 divisions que fem: per finestres, per intervals, per recordings i per pacients.
Totes per això tenen un element en comú, el tractament arxiu a arxiu amb un bucle *for*. Carreguem un arxiu, el tractem balancejant i/o fent la divisió train test, guardem les dades en format X (N,21,128) i Y(N) i passem al següent arxiu. Al final ens queden 4 llistes: Xtrain, Xtest, Ytrain, Ytest. Les X contenen les dades com a tal i la Y conte la classe. Per treballar el que fem és primer carregar el dataframe, treballar amb l'índexs d'aquests i per últim amb el índex resultant final carreguem les dades .npz.

- Window: Aquesta és la divisió més simple i més fàcil. Dintre del bucle per a cada arxiu carreguem un nombre de les dades de l'arxiu .npz i el mateix nombre d'etiquetes. En aquest cas del dataframe només utilitzem la columna de la classe per treure les etiquetes i per mirar que estem agafant el mateix nombre de les dues classes. Un cop ja hem recorregut els arxius i tenim les dades en dues llistes X i Y passem a dividir-les. Fem servir la funció train_test_split de la llibreria sklearn i ja ho tenim.

- Interval: Ara farem servir més el datafrem de les metadades. Dintre del bucle primer mirem quants intervals té el pacient (l'arxiu que toca ara). D'aquests agafem una porció anirà a test i la resta a train. Fem per tant dos "bucles" aquí dins on, en cada iteració busquem els índexs que pertanyen a l'interval en concret. D'aquests índexs fem un subsampling. Aquest serà equilibrat en les dues classes si anirà al train i serà aleatori si va al test.

- Recording: Dividir per gravació es tracta del mateix que per interval però canviant una variable. On abans posàvem "global_interval" ara posem "filename". De fet, en codi.py es tracta del mateix, ja que declarem una variable separadora que depenen de la divisió té un valor o un altre.

- Pacient: Aquesta divisió igual que la de les finestres és bastant simple. Això es deu al fet que ja recorrem cada pacient individualment. Per tant, l'únic que hem de fer és fet un subsampling aleatori dels primers 4 pacients i enviar-lo a train. I amb la resta fer un subsampling balancejat i enviar-lo a test.

Amb aquesta forma de dividir aconseguim solucionar els problemes de desbalanceig mencionats en la introducció. Sent la més perillosa i que hem evitat el fet d'estar desbalancejat en el conjunt d'entrenament respecte a les classes. Això hauria pogut portar al fet que els nostres models es centressin a predir més d'una classe que de l'altre.

A partir d'ara ens referirem al conjunt de Xtest i Ytest com a conjunt de validation. Això ho fem així per evitar confusions posteriorment.


### XARXA NEURONAL: ENCODER I LSTM

Ens hem basat en la següent estructura per fer la xarxa neuronal.
![image](https://github.com/YasminLH/psiv2_epilepsy/assets/101893393/4570c698-fa07-48bb-ae09-f00959730f8b)

La raó d'aquesta elecció ha estat perquè així podíem reutilitzar part del codi de l'anterior repte i perquè en comparació amb l'altra opció aquesta se'ns feia més fàcil d'implementar.


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
L'última capa conté dues neurones de sortida, ja que es tracta d'un problema de classificació binària (si epilèpsia  o no epilèpsia)
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

BCEWithLogitsLoss: Aquesta pèrdua combina una capa sigmoide i la BCELoss en una sola classe. Aquesta versió és més estable numèricament que utilitzar un sigmoide senzill seguit d'un BCELoss ja que, combinant les operacions en una sola capa, aprofitem el truc log-sum-exp per a l'estabilitat numèrica. El BCELoss és una mesura que l'entropia creuada binària entre les probabilitats d'entrada i de la sortida.



### LSTM

Una LSTM com a classificador té la capacitat de processar seqüències, aprendre dependències a llarg termini i generar una representació interna de les dades que permet realitzar tasques de classificació on hi ha presents patrons seqüencials en les dades d'entrada. Això fa que sigui de gran utilitat en el nostre cas on analitzem conjunt de dades que tracten sobre la visualització durant hores de senyals del servei on l'impuls actual depèn de l'impuls anterior. Si hi ha alguna seqüència o patró alhora d'obtenir l'atac el detectarem amb LSTM.

La xarxa LSTM contè la següent estructura:

    - LSTM - (input = 21, hidden =  20, output =2)
    - LSTM - (input = 21, hidden =  20, output =2)
    - Dropout - (0.5) 
    - Batch Normalization - (hidden = 20, output = 2)
    - Linear - (hidden = 20, output = 2)


Consta de dues capes de LSTM per aprendre patrons en les seqüències de dades que rep, seguides d'un Dropout i Batch Normalization, per reduir l'overfiting del nostre model i generalitzar al màxim, i així obtenir uns millors resultats del nostre model. El Dropout a 0.5, el que fa és inhibir a la meitat de les neurones a l'hora de l'entrenament i el batch normalization, s'encarrega d'estabilitzar i accelerar l'entrenament, normalitzant les activacions a cada capa.
Finalment, tenim una capa fully connected "Linear" que s'encarrega de la classificació binària, per saber si la persona té un atac d'epilèpsia o no.

La mètrica per avaluar el rendiment del model, és la precisió, ja que és la mètrica òptima i correcte en aquest context mèdic de classificació de pacients per epilèpsia.

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
Utilitzem  la BCEWithLogitsLoss, per calcular les discrepàncies entre les prediccions del model i les etiquetes reals i  obtenim la loss, específicament una loss per les dades train i una loss per les dades test. Això ho guardem en un objecte pickle, per tal de poder manipular més endavant com vulguem. El codi on generem els gràfics pertinents està al fitxer "pickle.py". Aquestes gràfiques ens serveixen per monitorar i veure quins dels models que hem generat ens va millor, generalitza més i evitar l'overfitting. 

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
Segueix-la més o menys la tendència òptima de la loss tant de train com de test.

Com a resultat final podem veure comparant les losses que la xarxa neuronal Encoder, va millor que la del LSTM, això podria ser que l'LSTM no estigui explotant de manera òptima les relacions temporals de les dades, ja que en un segon no arriba a haver-hi molt de canvi o que tingui una arquitectura molt més complexa i fa que li costi més aprendre de manera molt més precisa els patrons locals.
Per exemple l'encoder té capes específiques que aprenen patrons locals i capes que disminueixen la dimensionalitat, per tal d'eliminar la complexitat i deixar només aquelles característiques més importants.

** Les altres losses dels nivells recording, interval i pacient estan a la carpeta losses. **

Els millors resultats són els que s'han vist anteriorment, és a dir el que fa la divisió per window.

### Resultats Millor Model

Un cop hem vist quin dels models és més eficient, toca provar-lo amb el conjunt de validation. Aquest es tracta del conjunt de dades que havíem anomenat com a Xtest i Ytest en la part de dataloading. Per evitar confusions amb les gràfiques de kfold ens referim a aquest conjunt com a validation.
Agafem el model i amb una versió modificada de la funció test li passem les dades. Ens retornarà (realment ho guarda) tant la loss com els outputs del model. Amb aquests outputs podrem calcular l'accuracy.

Ara veurem una taula per encoder i lstm on trobem els resultats delsmillor model per cada una de les divisions. En aquesta taula trobem la divisió a la que pertanyen, la loss en train i validation, l'accuracy i si el model es troba esbiaixat cap a una de les classes, és a dir, busca predir més d'una classe ignorant a l'altre.
Considerem que un model està esbiaixat si la diferència entre els accuracys específics per cada classe és més gran de 0,05.


#### Resultats Encoder millor Model 

| Divisó  | Validation loss | Train loss  | Accuracy  | Esbiaixat |
| --------- | ------- | ------ | ----- | -- |
| Window    |  0.2414 | 0.0558 | 0.935 | No |
| Interval  |  0.6332 | 0.3317 | 0.744 | No |
| Recording |  0.4890 | 0.3441 | 0.810 | No |
| Pacient   |  0.6286 | 0.1890 | 0.822 | No |

Podem observar d'aquesta taula vàries coses. La primera és que cap model es troba segat, de fet la diferència màxima és del 0.003. També podem veure que obtenim els millors resultats amb la divisió window. Això era d'esperar ja que és la generalització més gran. Ca nivell de generalització té dades més diferents. Tot a això, trobem un clar outlier amb la divisió d'interval. Que en comptes de tenir un valor semblant al de recording té un valor més baix que aquest i que el de pacient. Això es deu segurament a què hi ha hagut més overfiting en aquesta divisió. Ho sabem perquè és on tenim la diferència més gran entre train loss i validation loss.
Deixant interval de banda veiem que tenim molt bons resultats.

#### Resultats LSTM millor Model 

| Divisó  | Validation loss | Train loss  | Accuracy  | Esbiaixat |
| --------- | ------- | ------ | ----- | -- |
| Window    |  0.6254 | 0.5509 | 0.713 | No |
| Interval  |  0.4949 | 0.4798 | 0.801 | No |
| Recording |  0.6749 | 0.6515 | 0.650 | No |
| Pacient   |  0.7093 | 0.5138 | 0.690 | No |

Podem veure com igual que en el model anterior no trobem que estigui esbiaixat.
També veiem que en Window, Recording i Pacient segueix el mateix patró que abans. Tenim els mateixos valors però per sota. En canvi, en interval passa el contrari, tenim un valor d'accuracy molt elevat. Podem veure que justament pasa el contrari en els seues valors de train loss i validation loss. No trobem res de overfiting. De fet, fins i tot en el pitjor resultat en accuracy, trobem molt menys overfiting en el lstm que en el encoder.

Creiem que això no es deu al LSTM com a tal sinó al fet que l'hem aturat abans que fes overfiting (amb menys èpoques), i que amb més èpoques no hagués millorat els seus resultats a diferència de l'autoencoder.

## CONCLUSIÓ

Podem concloure que hem aconseguit assolir tots els nostres objectius satisfactòriament, hem classificat binariament de manera correcte amb els 4 nivells i hem pogut veure que el nivell de window és pronunciadament més precís que els altres.
A més a més, evitant el desbalanceig de les dades, ja que els nostres models no es troben esbiaixats.
Per altra banda gràcies al K-Fold, hem pogut triar quin és el millor model dels 4 que s'han generat, destacant aquell com a més eficient en generalització i que resisteix l'overfitting.

Per altra banda, hem pogut veure que la xarxa neuronal Encoder va molt millor que la de l'LSTM, això és degut a que podria ser que l'LSTM no estigui explotant de manera òptima les relacions temporals de les dades, ja que en un segon no arriba a haver-hi molt de canvi o pot ser que tingui una arquitectura molt més complexa i fa que li costi més aprendre de manera molt més precisa els patrons locals.

Finalment, les dues divisions que més ens importen són la d'intervals i la recording, ja que són les que realisticament som més aplicables en la vida real.
Trobem que el model LSTM és bo en interval i encoder és bo en recording.
Depenent de la situació pot sortir a compte fer servir un o l'altre. En cas de dubte creiem que la millor opció seria l'encoder, ja que en interval no queda tant per darrere del lstm en accuracy.


