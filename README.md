# REPTA EPILEPSIA AMB ELECTROENCEFALOGRAMA


## INTRODUCCIÓ

Analitzarem diferentes senyals celebrals per tal entrenar un model d'inteligencia artificial que pugui detedectar si una persona està petint o no un atac d'epilepsia, per tal de definir quan una persona rep o no un atac s'utlitzaran finiestres. És a dir un grup de senyals serán agrupats determinan un temps en l'espai de la persona on estarà etiquetat amb no té epilepsia i amb té epilepsia. Amb aquestes finestres etiquetades s'hauran de tractar perque el espai de temps que representa es pogui tractar com un únic senyal i no com un conjunt de senyals i aixì poder entrenar el model. Ja que l'ordre del senyal té importancia ja que un conjunt de finiestres d'un mateix pacient estan continues en el mateix temps. La nostre red ha de aprnedre del pasat ja que per una mateixa finistra té diferents senyals on el senyals tenen un ordre temporal i un senyal del pasat influeix en els senyals següents. Primer hem decidit utlitzar encoders per fer la clasificació de finestres, ja que és més simple i aixì poder veure si la part de tractament d'imatge i creació de finestres s'ha pogut fer bé per tal de que la IA pogui entrenar del nostre model. Al realitzar auqest procés i obtindré els resultats pasarem a utilitzar una LSTM on el model amb millors resultats serà el qual ens quedarem. 

L'encoder cambiara el embedding de carecteristiques del senyal rebut per poder extreure les carecteritiques d'una manera en la qual siguin més fàcils de clasificar i poder diferenciar entre les diferents classes que rep el model.

En canvi la LSTM és un tipus de RNN, amb la capacitat recordar patrons a llarg termini. Ja que les RNN tradicionals sempre tenen problemes amb el gradient amb següencies llargues. Obtenint aixì aquesta habilitat amb celdes de memoria que estan regulades per celdes que controlen la informació que surt i entra a les celdes de memoria. FEnt aixì que pogui oblidar informació que no és importan i matenint tot aquella informació que aporta importancia i permet aprendre al model.

## OBJECTIUS

L'objectiu d'aquest treball és veure si podem veure si una persona està tenint un atac o no d'epilepsia, per tal de saber si tenim els recursos i la capcitat per determinar que una persona està patint un atac amb els senyals cerebrals.
També tenim com a objectiu tractar el gran desequilibri de les dades ja que per cada pacient hi han una gran quantitat de finestres on les persones no estan rebent un atac epilepsia i molt poques on si s'ha esta patint un atac. S'ha de tractar el dataset per equilibrar les dades o fer que aquest desequilibri no afecti al model.
Com hi ha diferetns maneres de definir que és  train o test és a dir que podem fer un train i test segons el pacients, finestres o  atacs d'epilepsia.Creant aixì com objectiu fer un model complert que defineixi millor quines són les metriques i com s'evaluarà el model perquè els resultats tinguin pes per prendre decisions. Complir amb aquests objectius en l'ambit médic és molt important ja que hi ha vides en joc i la presa d'una mala desició per una erronea interpretació del model pot ser molt costós pels pacients. Fent que hi hagi una base per aplicacions futeres médiques erronea perjudicant a principalment als pacients. 


## BASE DE DADES 


## DISTRIBUCIÓ


## PROCEDIMENT


### XARXA NEURONAL: ENCODER I CLASSIFICACIÓ


### LSTM


## RESULTATS


## CONCLUSIÓ





##
