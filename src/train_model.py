import argparse, os, yaml, joblib
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FEATURES = [
    'SignupDay','SignupTime','DeviceType','IsFreeConsultationTaken','CountryId','Gender','Age','City','State',
    'HasTakenChatConsultation','HasTakenCallConsultation','HasTakenOtherConsultation','HasRated','HasReviewed',
    'HadPositiveInteraction','HadNegativeInteraction','HasUsedGift','app_version','device_manufacturer','device_name',
    'isp','language','os_name','os_version','platform','store','lifetime_session_count','session_count','time_spent',
    'Acquisition Source','Campaign','Campaign Adgroup','Campaign Group'
]

def score_bucket(x):
    if x >= 0.80: return '0.80-1.00 Very High'
    if x >= 0.60: return '0.60-0.80 High'
    if x >= 0.40: return '0.40-0.60 Medium'
    if x >= 0.20: return '0.20-0.40 Low'
    return '0.00-0.20 Very Low'

def percentile_distribution(df, score_col, actual_col=None):
    out = df.copy()
    out['Percentile_Bucket'] = pd.qcut(out[score_col].rank(method='first'), 10, labels=[
        'Top 10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','Bottom 10%'
    ])
    agg = {'UserLoginId':'count', score_col:['min','max','mean']}
    if actual_col and actual_col in out.columns:
        agg[actual_col] = ['sum','mean']
    summary = out.groupby('Percentile_Bucket', observed=False).agg(agg).reset_index()
    summary.columns = ['_'.join([str(i) for i in c if i]) for c in summary.columns]
    return summary

def feature_level_score_bucket_analysis(df, features, score_col):
    rows=[]
    temp=df.copy()
    temp['Score_Bucket']=temp[score_col].apply(score_bucket)
    for feature in features:
        if feature not in temp.columns: continue
        if pd.api.types.is_numeric_dtype(temp[feature]):
            g=temp.groupby('Score_Bucket')[feature].agg(['count','mean','median']).reset_index()
            for _,r in g.iterrows():
                rows.append({'Feature':feature,'Score_Bucket':r['Score_Bucket'],'Metric':'numeric_mean','Value':round(r['mean'],4),'Median':round(r['median'],4),'Count':int(r['count'])})
        else:
            top=temp.groupby('Score_Bucket')[feature].agg(lambda x: x.value_counts().index[0] if len(x)>0 else None).reset_index()
            for _,r in top.iterrows():
                rows.append({'Feature':feature,'Score_Bucket':r['Score_Bucket'],'Metric':'top_category','Value':r[feature],'Median':'','Count':''})
    return pd.DataFrame(rows)

def main(config_path):
    with open(config_path) as f: cfg=yaml.safe_load(f)
    train_path=cfg['data']['train_path']; pred_path=cfg['data']['prediction_path']
    target=cfg['columns']['target']; user_id=cfg['columns']['user_id']
    output_dir=cfg['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True); os.makedirs(os.path.dirname(cfg['output']['model_path']), exist_ok=True)

    train_df=pd.read_csv(train_path)
    pred_df=pd.read_csv(pred_path)
    use_features=[c for c in FEATURES if c in train_df.columns and c in pred_df.columns]
    X=train_df[use_features].copy(); y=train_df[target].copy(); Xp=pred_df[use_features].copy()

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=cfg['model']['test_size'],random_state=cfg['model']['random_state'],stratify=y)
    num_cols=X_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols=[c for c in use_features if c not in num_cols]

    medians=X_train[num_cols].median()
    for frame in [X_train,X_test,Xp]:
        frame[num_cols]=frame[num_cols].fillna(medians)
        for c in cat_cols:
            frame[c]=frame[c].fillna('unknown').astype(str).str.lower().str.strip()

    encoders={}
    for c in cat_cols:
        le=LabelEncoder()
        le.fit(pd.concat([X_train[c],X_test[c],Xp[c]]).astype(str).unique())
        X_train[c]=le.transform(X_train[c]); X_test[c]=le.transform(X_test[c]); Xp[c]=le.transform(Xp[c])
        encoders[c]=le

    model=lgb.LGBMClassifier(objective='binary', n_estimators=cfg['model']['n_estimators'], learning_rate=cfg['model']['learning_rate'], max_depth=cfg['model']['max_depth'], num_leaves=cfg['model']['num_leaves'], random_state=cfg['model']['random_state'], n_jobs=-1, importance_type='gain')
    model.fit(X_train,y_train,eval_set=[(X_test,y_test)],eval_metric='auc',callbacks=[lgb.early_stopping(50, verbose=False)])

    train_probs=model.predict_proba(X_train)[:,1]; test_probs=model.predict_proba(X_test)[:,1]; pred_probs=model.predict_proba(Xp)[:,1]
    metrics=pd.DataFrame([{
        'AUC':roc_auc_score(y_test,test_probs),'Accuracy_0.5':accuracy_score(y_test,(test_probs>=.5).astype(int)),
        'Precision_0.5':precision_score(y_test,(test_probs>=.5).astype(int),zero_division=0),
        'Recall_0.5':recall_score(y_test,(test_probs>=.5).astype(int),zero_division=0),
        'F1_0.5':f1_score(y_test,(test_probs>=.5).astype(int),zero_division=0)
    }])

    feature_imp=pd.DataFrame({'Feature':use_features,'Importance':model.feature_importances_})
    feature_imp['Importance_Percentage']=(feature_imp['Importance']/feature_imp['Importance'].sum()*100).round(2)
    feature_imp=feature_imp.sort_values('Importance_Percentage',ascending=False)

    train_out=train_df.loc[X_train.index,[user_id,target]].copy(); train_out['score']=train_probs
    test_out=train_df.loc[X_test.index,[user_id,target]].copy(); test_out['score']=test_probs
    pred_out=pred_df[[user_id]].copy(); pred_out['score']=pred_probs; pred_out['Score_Bucket']=pred_out['score'].apply(score_bucket)

    train_pct=percentile_distribution(train_out,'score',target)
    test_pct=percentile_distribution(test_out,'score',target)
    pred_pct=percentile_distribution(pred_out,'score',None)

    pred_full=pred_df.copy(); pred_full['score']=pred_probs
    feature_bucket=feature_level_score_bucket_analysis(pred_full,use_features,'score')

    feature_imp.to_csv(os.path.join(output_dir,'feature_importance.csv'),index=False)
    train_pct.to_csv(os.path.join(output_dir,'train_percentile_distribution.csv'),index=False)
    test_pct.to_csv(os.path.join(output_dir,'test_percentile_distribution.csv'),index=False)
    pred_pct.to_csv(os.path.join(output_dir,'prediction_percentile_distribution.csv'),index=False)
    feature_bucket.to_csv(os.path.join(output_dir,'feature_level_score_bucket_analysis.csv'),index=False)
    pred_out.sort_values('score',ascending=False).to_csv(os.path.join(output_dir,'all_predicted_users.csv'),index=False)
    metrics.to_csv(os.path.join(output_dir,'model_metrics.csv'),index=False)

    with pd.ExcelWriter(os.path.join(output_dir,'Model_Analysis_Report.xlsx'),engine='openpyxl') as writer:
        metrics.to_excel(writer,'Model_Metrics',index=False)
        feature_imp.to_excel(writer,'Feature_Importance',index=False)
        train_pct.to_excel(writer,'Train_Percentiles',index=False)
        test_pct.to_excel(writer,'Test_Percentiles',index=False)
        pred_pct.to_excel(writer,'Prediction_Percentiles',index=False)
        feature_bucket.to_excel(writer,'Feature_Bucket_Analysis',index=False)
        pred_out.to_excel(writer,'All_Predictions',index=False)

    joblib.dump({'model':model,'features':use_features,'encoders':encoders,'medians':medians}, cfg['output']['model_path'])
    print(f'Done. Outputs saved to {output_dir}')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config.yaml')
    args=parser.parse_args()
    main(args.config)
