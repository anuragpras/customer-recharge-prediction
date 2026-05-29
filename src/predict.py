import yaml, joblib, pandas as pd, os

def main(config_path='config.yaml'):
    with open(config_path) as f:
        cfg=yaml.safe_load(f)
    bundle=joblib.load(cfg['output']['model_path'])
    model=bundle['model']; features=bundle['features']; encoders=bundle['encoders']; medians=bundle['medians']
    df=pd.read_csv(cfg['data']['prediction_path'])
    X=df[features].copy()
    num_cols=list(medians.index)
    X[num_cols]=X[num_cols].fillna(medians)
    for c,le in encoders.items():
        X[c]=X[c].fillna('unknown').astype(str).str.lower().str.strip()
        X[c]=X[c].where(X[c].isin(le.classes_), le.classes_[0])
        X[c]=le.transform(X[c])
    df['score']=model.predict_proba(X)[:,1]
    os.makedirs(cfg['output']['output_dir'], exist_ok=True)
    df[[cfg['columns']['user_id'],'score']].sort_values('score', ascending=False).to_csv(os.path.join(cfg['output']['output_dir'],'prediction_only.csv'),index=False)
    print('Prediction file created.')

if __name__ == '__main__':
    main()
