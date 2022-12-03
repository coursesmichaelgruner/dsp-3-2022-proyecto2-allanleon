 pkg load statistics


 "Setting the chart properties"
 function confusion_matrix(Yt, Yp, commands)
    % cm=confusionchart (Yt, Yp, "Title", ...
    % "Demonstration with summaries","Normalization",...
    % "absolute","ColumnSummary", "column-normalized","RowSummary",...
    % "row-normalized");
    % printf('hola')
    commands=char(commands);

    cm = confusionmat(Yt,Yp);
    cc = confusionchart(cm, commands, ...
        "Title","Confusion Matrix for Validation Data", ...
        "ColumnSummary","column-normalized","RowSummary","row-normalized");
    labels = ['down'; 'go'; 'left'; 'no'; 'off';'on'; 'right'; 'stop'; 'up'; 'yes'; 'unknown';'noise'];

    sortClasses(cc,[labels])
    waitforbuttonpress()
endfunction
