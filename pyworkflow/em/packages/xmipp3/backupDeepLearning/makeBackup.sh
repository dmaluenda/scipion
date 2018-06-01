
TARGETPATH="/home/rsanchez/app/scipion/pyworkflow/em/packages/xmipp3/backupDeepLearning"
NUM_NEXT_BACKUP=`ls $TARGETPATH | grep ^v | wc -l`
if [ "$1" != same ]
then
  NUM_NEXT_BACKUP=$(($NUM_NEXT_BACKUP  + 1 ))
  NEWPATH=$TARGETPATH/v$NUM_NEXT_BACKUP
  mkdir $NEWPATH
else
  echo "Overriding backup dir"
  NEWPATH=$TARGETPATH/v$NUM_NEXT_BACKUP
fi

echo  $NEWPATH
cp /home/rsanchez/app/scipion/pyworkflow/em/packages/xmipp3/*deep* $NEWPATH
cp /home/rsanchez/app/scipion/pyworkflow/em/packages/xmipp3/networkDef.py $NEWPATH
