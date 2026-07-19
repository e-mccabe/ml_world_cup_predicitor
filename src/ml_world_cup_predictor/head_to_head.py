from ml_world_cup_predictor.utils import get_long_team_df 

def generate_head_to_head_pct(df,group_columns,agg_col,suffix = 'group'):

    #df = df.reset_index(names = 'match_id').copy()

    long = get_long_team_df(df)

    df_with_features = _get_grouped_features(df,long,group_columns,agg_col,suffix)

    return df_with_features


def _get_grouped_features(df,long_df,group_columns,agg_col,suffix):

    
    grouped = long_df.groupby(group_columns)[agg_col]

    long_df['group_col'] = (grouped.cumsum() - long_df[agg_col])/grouped.cumcount()

    for side in ['home','away']:
        values = long_df.loc[long_df['side'] == side].set_index('match_id')['group_col']
        df[f'{side}_{suffix}'] = values.to_numpy()

    return df
