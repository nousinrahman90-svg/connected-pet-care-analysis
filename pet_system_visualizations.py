"""Reproducible shelter evidence dashboard. See README.md for interpretation."""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(animal_path, sac_path, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(animal_path)
    df = raw.drop_duplicates().copy()
    for c in ['intakedate', 'movementdate', 'returndate', 'deceaseddate']:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    adopted = df[df.movementtype.eq('Adoption')].copy()
    adopted['days_to_return'] = (adopted.returndate - adopted.movementdate).dt.days
    adopted['recorded_return'] = adopted.returndate.notna() & adopted.days_to_return.ge(0)
    returns = adopted[adopted.recorded_return].copy()
    # Reasons on NON-returned rows are not evidence of a return.
    returns['returnedreason'] = returns.returnedreason.fillna('Not recorded')
    sac_raw = pd.read_csv(sac_path, header=1, encoding='utf-8-sig', dtype=str)
    # SAC uses two species blocks with duplicate column names. Position preserves meaning.
    assert len(sac_raw.columns) == 35, 'Unexpected SAC schema: expected State/Year/Org Count + two 16-metric species blocks'
    parts = []
    for species, start in [('Dog', 3), ('Cat', 19)]:
        block = sac_raw.iloc[:, start:start+16].copy()
        block.columns = [c.split('.')[0] for c in block.columns]
        for c in block:
            block[c] = pd.to_numeric(block[c].str.replace(',', '', regex=False), errors='coerce')
        block.insert(0, 'species', species)
        block.insert(0, 'year', pd.to_numeric(sac_raw.Year, errors='raise'))
        block.insert(0, 'state', sac_raw.State)
        block['org_count'] = pd.to_numeric(sac_raw['Org Count'].str.replace(',', '', regex=False), errors='coerce')
        parts.append(block)
    sac = pd.concat(parts, ignore_index=True)
    assert not sac.duplicated(['state','year','species']).any(), 'Duplicate SAC keys'
    sac.to_csv(out/'sac_state_year_species.csv', index=False)
    # Combine at metric-summary level, never join incompatible individual/state grains.
    rows = []
    for (year, species), group in adopted.groupby([adopted.movementdate.dt.year, 'speciesname']):
        for metric, count in [('Adoption events',len(group)),('Recorded adoption return events',int(group.recorded_return.sum()))]:
            rows.append(dict(source='animal-data-1',grain='movement-year/species', geography='Unspecified shelter coverage',year=year,species=species,metric=metric,count=count))
    for _, row in sac.iterrows():
        for metric in sac.columns[3:-1]:
            rows.append(dict(source='SAC 2024–2025',grain='state/year/species',geography=row.state,year=row.year,species=row.species,metric=metric,count=row[metric]))
    pd.DataFrame(rows).to_csv(out/'combined_metrics.csv',index=False)
    # Avoid exporting names, IDs and microchip numbers into the public evidence package.
    reasons = returns.returnedreason.value_counts()
    reasons.rename_axis('recorded_reason').reset_index(name='return_events').to_csv(out/'adoption_return_reasons.csv',index=False)
    species = adopted.groupby('speciesname').agg(adoption_events=('id','size'),recorded_return_events=('recorded_return','sum'))
    species['recorded_return_pct'] = 100*species.recorded_return_events/species.adoption_events
    species.to_csv(out/'adoption_returns_by_species.csv')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    def save(fig,name,foot='Source: supplied files. Recorded events; descriptive evidence, not causal effects.'):
        fig.text(.02,.015,foot,fontsize=9,color='#555555')
        fig.tight_layout(rect=[0,.055,1,.95]); fig.savefig(out/(name+'.png'),dpi=170); plt.close(fig)
    def bars(series,title,name,xlabel='Recorded events'):
        fig,ax=plt.subplots(figsize=(12,max(5,len(series)*.34)))
        series.sort_values().plot.barh(ax=ax,color='#167d9a')
        ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel(xlabel);ax.set_ylabel('')
        for patch in ax.patches: ax.annotate(f'{patch.get_width():,.0f}',(patch.get_width(),patch.get_y()+patch.get_height()/2),xytext=(4,0),textcoords='offset points',va='center',fontsize=9)
        ax.set_xlim(0, max(series.max()*1.18,1));save(fig,name)
    bars(reasons,'Why adopted animals returned: all recorded reasons','01_return_reasons')
    bars(species.adoption_events,'Adoption events by species','02_adoptions_by_species')
    bars(species.recorded_return_pct,'Recorded return share of adoption events by species','03_return_share_by_species','Recorded return events / adoption events (%)')
    # Timing uses return-event denominators, not assumed complete follow-up.
    bins = pd.cut(returns.days_to_return,[-1,7,30,90,180,365,np.inf],labels=['0–7 days','8–30 days','31–90 days','91–180 days','181–365 days','Over 365 days'])
    timing = bins.value_counts(sort=False);timing.to_csv(out/'return_timing.csv')
    bars(timing,'When recorded adoption returns occurred','04_return_timing')
    annual=adopted.groupby(adopted.movementdate.dt.year).agg(adoptions=('id','size'),returns=('recorded_return','sum'))
    fig,ax=plt.subplots(figsize=(11,5)); annual.plot.bar(ax=ax,color=['#167d9a','#e99544']);ax.set_title('Adoption-year cohorts and recorded returns');ax.set_ylabel('Events');ax.set_xlabel('Adoption year')
    save(fig,'05_adoption_cohorts','Different follow-up durations and incomplete years: do not interpret this as a return-risk trend.')
    repeat=returns.groupby('id').size().value_counts().sort_index()
    bars(repeat.rename(index=lambda x:f'{x} recorded return(s)'), 'Repeated adoption returns: distinct animals','06_repeat_returns','Distinct animals')
    national=sac.groupby(['year','species']).sum(numeric_only=True)
    outcomes=['Adoption','Return to Owner','Return to Field','Transferred Out','Other Live Outcomes','Shelter Euthanasia','Died in Care','Lost in Care']
    fig,ax=plt.subplots(figsize=(13,6));national[outcomes].plot.bar(stacked=True,ax=ax,colormap='tab20');ax.set_ylabel('Reported outcome events');ax.set_title('SAC reported outcomes across supplied jurisdictions');ax.set_xlabel('Year / species');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=9)
    save(fig,'07_sac_outcomes','SAC supplied jurisdictions; reporting coverage may differ. Return to Owner is NOT an adoption return.')
    ny=sac[sac.state.eq('NY')].set_index(['year','species'])
    fig,ax=plt.subplots(figsize=(12,6));ny[['Gross Intakes','Adoption','Shelter Euthanasia','Died in Care']].plot.bar(ax=ax,color=['#167d9a','#72a981','#e99544','#bf5366']);ax.set_title('New York: reported intake and care-related outcomes');ax.set_ylabel('Events');ax.set_xlabel('Year / species');save(fig,'08_new_york')
    intake=['Stray at Large','Relinquished by Owner','Seized','Other Intakes','Transferred In']
    fig,ax=plt.subplots(figsize=(12,6));national[intake].plot.bar(stacked=True,ax=ax,colormap='Set2');ax.set_title('Shelter intake pathways: support needs before and after intake');ax.set_ylabel('Reported intake events');ax.set_xlabel('Year / species');save(fig,'09_intake_pathways')
    # Adoption movement minus intake is a proxy, not proven uninterrupted shelter stay.
    adopted['intake_to_adoption_days']=(adopted.movementdate-adopted.intakedate).dt.days
    valid=adopted[adopted.intake_to_adoption_days.ge(0) & adopted.speciesname.isin(['Dog','Cat'])]
    fig,ax=plt.subplots(figsize=(10,5));data=[valid.loc[valid.speciesname.eq(s),'intake_to_adoption_days'] for s in ['Dog','Cat']];ax.boxplot(data,tick_labels=['Dog','Cat'],showfliers=False);ax.set_title('Recorded intake-to-adoption interval');ax.set_ylabel('Days (outliers hidden)');save(fig,'10_care_interval','Proxy interval includes possible foster/other movements; not continuous shelter length of stay.')
    summary={'input_rows':len(raw),'exact_duplicate_rows_removed':len(raw)-len(df),'adoption_events':len(adopted),'recorded_return_events':len(returns),'distinct_animals_returned':int(returns.id.nunique()),'recorded_return_share_pct':round(100*len(returns)/len(adopted),2),'invalid_return_date_order':int((adopted.days_to_return<0).sum()),'movement_date_min':str(df.movementdate.min()),'movement_date_max':str(df.movementdate.max()),'return_date_max':str(df.returndate.max()),'returned_within_30_days':int(returns.days_to_return.le(30).sum()),'reasons':reasons.to_dict()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--animals',default='animal-data-1.csv');parser.add_argument('--sac',default='sac_2yr_aggregate_24_25.csv');parser.add_argument('--output',default='outputs');args=parser.parse_args();run(args.animals,args.sac,args.output)
