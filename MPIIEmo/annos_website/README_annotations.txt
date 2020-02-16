This text provides information on how to use the annotations of the MPIIEmo dataset.

The file raw_annotations.mat contains a struct with the following fields:

- scenario     : ID of the scenario
- subscenario  : ID of the subscenario
- actorA       : ID of actor starting in kitchen
- actorB       : ID of actor entering kitchen
- ratedActor   : indicates, to which actor the ratings belong
- videoTime    : the corresponding time in the video, to which the rating belongs (in units of 100ms; sampling rate was 10hz)

- {Anger,Happiness,Sadness,Surprise}      : ratings for the categorial emotion model
- {Activation,Anticipation,Power,Valence} : rating for the dimensional emotion model

The ratings are the raw annotations from each of the 5 annotators (e.g. compare size of annos.Anger).
They are not normalized and outliers are not removed.
You might want to do this and apply a method of your choice to aggregate the data from different annotators.

The rated actor can be identified in the video with the help of the portraits in the folder "actor_ids".

If you have any questions, feel free to contact pmueller@mpi-inf.mpg.de.
