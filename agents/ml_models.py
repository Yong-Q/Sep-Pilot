"""Bounded numeric GBR/RF/Ridge models; no pickle import or backend fitting.

Training and sklearn imports are called only by the allocation-bound worker.
Prediction and model inspection use validated JSON numeric arrays.
"""
import math
import struct


def _finite(value):
    if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value):
        raise ValueError('finite model numeric data required')
    return float(value)


def _sample(model,features):
    medians=model.get('medians')
    if not isinstance(medians,list)or not 1<=len(medians)<=256 or len(features)!=len(medians):
        raise ValueError('bounded model feature shape required')
    return [_finite(medians[i]if value is None else value)for i,value in enumerate(features)]


def _tree_values(trees,sample):
    if not isinstance(trees,list)or not 1<=len(trees)<=256:raise ValueError('bounded tree ensemble required')
    try:sample=[struct.unpack('f',struct.pack('f',value))[0]for value in sample]
    except OverflowError:raise ValueError('tree input exceeds float32 range')from None
    if not all(math.isfinite(value)for value in sample):raise ValueError('finite tree inputs required')
    output=[];total=0
    for tree in trees:
        arrays=[tree[key]for key in ('left','right','feature','threshold','value')]
        n=len(arrays[0]);total+=n
        if not n or total>200000 or any(len(array)!=n for array in arrays):
            raise ValueError('bounded tree arrays required')
        node=0
        for _ in range(n):
            if type(node)is not int or not 0<=node<n:raise ValueError('invalid tree child index')
            left,right,feature,threshold,value=(array[node]for array in arrays)
            value=_finite(value)
            if left==right==-1:output.append(value);break
            if type(feature)is not int or not 0<=feature<len(sample):raise ValueError('invalid tree feature index')
            node=left if sample[feature]<=_finite(threshold)else right
        else:raise ValueError('cyclic tree rejected')
    return output


def predict(model,features):
    kind=model.get('format')if isinstance(model,dict)else None
    if kind not in('native-rf-json-v1','native-gbr-json-v1','native-ridge-json-v1'):
        raise ValueError('untrusted legacy model format; joblib import is disabled')
    sample=_sample(model,features)
    if kind=='native-rf-json-v1':
        values=_tree_values(model['trees'],sample)
        if len(values)<2:raise ValueError('RF uncertainty needs at least two trees')
        mean=sum(values)/len(values)
        spread=math.sqrt(sum((value-mean)**2 for value in values)/(len(values)-1))
        if not math.isfinite(mean)or not math.isfinite(spread):raise ValueError('nonfinite ensemble prediction')
        return mean,spread
    if kind=='native-gbr-json-v1':
        rate=_finite(model['learning_rate'])
        if not 0<rate<=1:raise ValueError('bounded GBR learning rate required')
        value=_finite(model['initial'])+rate*sum(_tree_values(model['trees'],sample))
        if not math.isfinite(value):raise ValueError('nonfinite model prediction')
        return value,None
    if kind=='native-ridge-json-v1':
        mean,scale,coef=(model[name]for name in ('mean','scale','coefficients'))
        if any(not isinstance(values,list)or len(values)!=len(sample)for values in (mean,scale,coef)):
            raise ValueError('bounded Ridge coefficient/scaler shape required')
        transformed=[]
        for value,center,divisor in zip(sample,mean,scale):
            center,divisor=_finite(center),_finite(divisor)
            if divisor<=0:raise ValueError('positive Ridge training scale required')
            transformed.append((value-center)/divisor)
        value=_finite(model['intercept'])+sum(value*_finite(weight)for value,weight in zip(transformed,coef))
        if not math.isfinite(value):raise ValueError('nonfinite model prediction')
        return value,None
    raise ValueError('untrusted legacy model format; joblib import is disabled')


def _tree(tree):
    return {'left':tree.children_left.tolist(),'right':tree.children_right.tolist(),
        'feature':tree.feature.tolist(),'threshold':tree.threshold.tolist(),
        'value':[float(value[0][0])for value in tree.value]}


def train(data,cpus,model_type='RF'):
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    kind=str(model_type).upper()
    if kind not in {'RF','GBR','RIDGE'}:raise ValueError('model_type must be GBR, RF, or Ridge')
    ids=[sample for sample in data['split']['train']if sample in data['labels']]
    if len(ids)<4:raise ValueError('at least four source-bound revealed TRAIN labels required')
    X=np.array([[float('nan')if value is None else value for value in data['rows'][sample]['features']]for sample in ids],dtype=float)
    raw_X=X.copy()
    medians=np.nanmedian(X,axis=0)
    if not np.isfinite(medians).all():raise ValueError('a descriptor has no observed TRAIN values')
    X=np.where(np.isnan(X),medians,X);y=np.array([data['labels'][sample]['value']for sample in ids],dtype=float)
    if len(set(y))<2:raise ValueError('TRAIN target has no variance')
    model={'features':data['features'],'target':data['target'],'unit':data['unit'],'medians':medians.tolist()}
    if kind=='RF':
        estimator=RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42,n_jobs=cpus)
        estimator.fit(X,y)
        model.update(format='native-rf-json-v1',trees=[_tree(member.tree_)for member in estimator.estimators_])
        importance=estimator.feature_importances_;interpretation='RF impurity-based importance, not causal evidence'
    elif kind=='GBR':
        estimator=GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=42)
        estimator.fit(X,y)
        model.update(format='native-gbr-json-v1',initial=float(estimator.init_.constant_[0][0]),
            learning_rate=float(estimator.learning_rate),trees=[_tree(member[0].tree_)for member in estimator.estimators_])
        importance=estimator.feature_importances_;interpretation='GBR impurity-based importance, not causal evidence'
    else:
        scaler=StandardScaler().fit(X);estimator=Ridge(alpha=1.0).fit(scaler.transform(X),y)
        model.update(format='native-ridge-json-v1',mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
            coefficients=estimator.coef_.tolist(),intercept=float(estimator.intercept_))
        importance=np.abs(estimator.coef_);interpretation='Absolute coefficients after TRAIN-only standard scaling, not causal evidence'
    # Validate serialization against the actual fitted estimator on TRAIN and
    # VALID inputs. TEST values are neither read nor used for model selection.
    valid=[sample for sample in data['split']['valid']if sample in data['labels']]
    probes=list(X[:32])+[np.array([medians[i]if value is None else value for i,value in enumerate(data['rows'][sample]['features'])])for sample in valid[:32]]
    for row in probes:
        actual=float(estimator.predict(scaler.transform(row.reshape(1,-1))if kind=='RIDGE'else row.reshape(1,-1))[0])
        encoded=predict(model,row.tolist())[0]
        if not math.isclose(actual,encoded,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('numeric model serialization differs from fitted estimator')
    errors=[predict(model,data['rows'][sample]['features'])[0]-data['labels'][sample]['value']for sample in valid]
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold,cross_validate
    folds=min(5,max(2,len(ids)//2))
    cv_estimator={'RF':lambda:RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42,n_jobs=cpus),
        'GBR':lambda:GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=42),
        'RIDGE':lambda:Ridge(alpha=1.0)}[kind]()
    steps=[('imputer',SimpleImputer(strategy='median'))]
    if kind=='RIDGE':steps.append(('scaler',StandardScaler()))
    steps.append(('model',cv_estimator))
    scores=cross_validate(Pipeline(steps),raw_X,y,
        cv=KFold(n_splits=folds,shuffle=True,random_state=42),
        scoring={'mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','r2':'r2'},
        n_jobs=1,error_score=float('nan'))
    def mean_finite(values,sign=1):
        finite=[float(value)*sign for value in values if math.isfinite(float(value))]
        return sum(finite)/len(finite)if finite else None
    facts={'n_train':len(ids),'train_ids':ids,'valid_ids':valid,
        'valid_mae':sum(abs(error)for error in errors)/len(errors)if errors else None,
        'valid_rmse':math.sqrt(sum(error*error for error in errors)/len(errors))if errors else None,
        'cv_folds':folds,'cv_mae_mean':mean_finite(scores['test_mae'],-1),
        'cv_rmse_mean':mean_finite(scores['test_rmse'],-1),'cv_r2_mean':mean_finite(scores['test_r2']),
        'cv_finite_folds':sum(math.isfinite(float(value))for value in scores['test_mae']),
        'cv_policy':'Seeded TRAIN-only KFold; imputer/scaler refitted inside every fold. VALID and TEST never fit preprocessing.',
        'test_ids_evaluated':[],'test_labels_used_for_training_or_acquisition':False,'model_type':kind,'seed':42,
        'labels_independently_verified':data['labels_independently_verified'],'label_source_kind':data['label_source_kind'],
        'synthetic_fixture':data['synthetic_fixture'],
        'n_valid':len(valid),'n_test':len(data['split']['test']),
        'test_metrics_status':'withheld_until_explicit_final_evaluation',
        'feature_importance':sorted([{'feature':name,'importance':float(value)}for name,value in zip(data['features'],importance)],key=lambda row:-row['importance']),
        'feature_importance_interpretation':interpretation,
        'uncertainty_interpretation':'ensemble_spread_proxy_not_calibrated'if kind=='RF'else'not_available_for_this_model'}
    return model,facts
