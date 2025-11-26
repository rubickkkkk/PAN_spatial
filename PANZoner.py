# coding=utf-8

import warnings
warnings.filterwarnings("ignore")
import scanpy as sc
import pandas as pd
import SOAPy_st as sp
import matplotlib.pyplot as plt
import squidpy as sq
import numpy as np
from scipy.sparse import csr_matrix
from anndata import AnnData
import os
import shutil


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:              
		os.makedirs(path)            
	else:
		print("---  folder existed  ---")

def trans2cell(adata,celltype:list):
    celltype_df = adata.obs[celltype]
    adata_celltype = AnnData(celltype_df)
    adata_celltype.uns = adata.uns
    adata_celltype.obsm = adata.obsm
    adata_celltype.obs = adata.obs[['in_tissue', 'array_row', 'array_col', 'patient', 'sample', 'location', 'disease_status', 'pri_organ', 'cancer_type', 'treatment', 'nd/r', 'number', 'dataset', 'batch']]
    return adata_celltype
    
    
def QC(adata,save_dir):
    sc.pp.filter_genes(adata, min_cells=10)
        
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
    # adata = adata[adata.obs['pct_counts_mt'] <= 25]
    adata = adata[adata.obs['pct_counts_in_top_50_genes'].dropna().index].copy()
    adata.obs['remove_label'] = 'not'
    adata.obs.loc[adata.obs['total_counts'] < 1000, 'remove_label'] = 'remove'
    with plt.rc_context():
        sc.pl.spatial(adata, cmap='magma',color=['remove_label'],ncols=4, size=1.3,img_key='hires',show=False)
        plt.savefig(f"{save_dir}/remove.pdf", dpi=80, bbox_inches='tight')
        plt.close()
    adata = adata[adata.obs['remove_label'] == 'not']
    return adata
    
    
def replace_nan_and_check(adata):
    from scipy.sparse import issparse
    if issparse(adata.X):
        #          ϡ     ʹ  .data   ʷ   Ԫ أ    滻NaNֵΪ0
        adata.X.data = np.nan_to_num(adata.X.data, nan=0)
        contains_na = np.isnan(adata.X.data).any()
    else:
        #           ܼ     ֱ  ʹ  np.nan_to_num
        adata.X = np.nan_to_num(adata.X, nan=0)
        contains_na = np.isnan(adata.X).any()
    
    print(f"adata.X contains NaN values: {contains_na}")
    return adata


def set_region(adata,var_added:str,str_contain:str,cut_off:int,save_dir):
    adata.var[var_added] = adata.var_names.str.contains(str_contain)
    sc.pp.calculate_qc_metrics(adata, qc_vars=[var_added], inplace=True, log1p=False,percent_top=(3,4))
    adata.obs['cell_type'] = 'Other'
    adata.obs.loc[adata.obs['pct_counts_Hypo'] > cut_off, 'cell_type'] = 'Hypo'
    with plt.rc_context():
        sc.pl.spatial(adata, cmap='magma',color=['cell_type'],ncols=4, size=1.3,img_key='hires',show=False)
        plt.savefig(f"{save_dir}/region_label.pdf", dpi=80, bbox_inches='tight')
        plt.close()
    return adata
    

def define_neighbors(adata,based_label:str,to_label:str,save_dir):
    """
    example: adata,neighbors = define_neighbors(adata,'Hypo','not_Hypo',save_dir)
    """
    sq.gr.spatial_neighbors(adata)
    neighbors = adata.obsp['spatial_connectivities']
    #   ȡϸ      ΪHypo  ϸ      
    hypo_cells = adata.obs.index[adata.obs['cell_type'] == based_label]

    #   ʼ  һ   µ  У    ڴ洢ÿ  Hypoϸ    core/border״̬
    adata.obs['cell_status'] = to_label

    for cell in hypo_cells:
        cell_index = adata.obs.index.get_loc(cell)
        neighbor_indices = neighbors[cell_index]
        row_indices, col_indices = neighbor_indices.nonzero()
        hypo_neighbors_count = (adata[col_indices].obs['cell_type'] == based_label).sum()
        if hypo_neighbors_count == 6 :
            adata.obs.at[cell, 'cell_status'] = 'core'
        elif hypo_neighbors_count ==5 :
            adata.obs.at[cell, 'cell_status'] = 'peri_core'
        elif hypo_neighbors_count >= 3 and hypo_neighbors_count <= 4 :
            adata.obs.at[cell, 'cell_status'] = 'border'
        else :
            adata.obs.at[cell, 'cell_status'] = 'isolated'

    with plt.rc_context():
        sc.pl.spatial(adata, cmap='magma',color=['cell_status'],ncols=4, size=1.3,img_key='hires',show=False)
        plt.savefig(f"{save_dir}/neighbor.pdf", dpi=80, bbox_inches='tight')
        plt.close()
    return adata,neighbors





def get_column_index(adata: AnnData, column_name: str) -> int:
    """
      ȡAnnData      `obs` DataFrame  ĳһ е       

        :
    adata (AnnData): AnnData    
    column_name (str): Ҫ                

        :
    int:    е       
    """
    try:
        return adata.obs.columns.get_loc(column_name)
    except KeyError:
        raise ValueError(f"Column '{column_name}' not found in AnnData object.")

def label_hypo_regions(adata: AnnData, neighbor_matrix: csr_matrix, based_label: str, based_labels: list, to_label: str) -> AnnData:
    """
         ṩ   ھӾ     AnnData     б          

        :
    adata (AnnData):       ϸ     ݵ AnnData    
    neighbor_matrix (csr_matrix):   ʾϸ     ھӹ ϵ  ϡ     
    based_label (str):     ȷ  ϸ  ״̬        
    based_labels (list):     ȷ  ϸ  ״̬       е ϸ     ͡ 
    to_label (str):    ڴ洢     ǵ       

        :
    AnnData:   `obs` DataFrame       'Hypo_region' е AnnData    
    
    example: adata,neighbor_list = label_hypo_regions(adata, neighbors, 'cell_status', ['core', 'peri_core'], 'Hypo_region')
    """
    region_count = 0
    adata.obs[to_label] = np.nan  #   ʼ  һ   µ      ڴ洢      

    #    ھӾ   ת  Ϊ ڽ  б 
    neighbor_list = {i: set() for i in range(neighbor_matrix.shape[0])}
    coo_matrix = neighbor_matrix.tocoo()
    for i, j in zip(coo_matrix.row, coo_matrix.col):
        if i != j:  #        ھӣ  Խ  ߣ 
            neighbor_list[i].add(j)
    
    def dfs(cell: int, region_label: str, based_label_idx: int, to_label_idx: int):
        """
                     DFS   Ա   ڽ ͼ е          

            :
        cell (int):   ʼϸ        
        region_label (str):   ǰ    ı ǡ 
        based_label_idx (int):     ȷ  ϸ  ״̬   е       
        to_label_idx (int):    ڴ洢     ǵ  е       
        """
        stack = [cell]
        while stack:
            current = stack.pop()
            if pd.isna(adata.obs.iat[current, to_label_idx]):
                adata.obs.iat[current, to_label_idx] = region_label
                for neighbor in neighbor_list[current]:
                    if adata.obs.iat[neighbor, based_label_idx] in based_labels and pd.isna(adata.obs.iat[neighbor, to_label_idx]):
                        stack.append(neighbor)

    #   ȡ    е     
    based_label_idx = get_column_index(adata, based_label)
    to_label_idx = get_column_index(adata, to_label)

    #     ÿ  ϸ           
    for cell in range(adata.shape[0]):
        if adata.obs.iat[cell, based_label_idx] in based_labels and pd.isna(adata.obs.iat[cell, to_label_idx]):
            region_count += 1
            region_label = f'Hypo_region_{region_count}'
            dfs(cell, region_label, based_label_idx, to_label_idx)

    return adata,neighbor_list



def label_border_cells(adata,to_label:str,based_label:str,based_labels:list , neighbor_list:dict ,save_dir:str):
    """
    example: adata = label_border_cells(adata,'Hypo_region','cell_status','border',neighbor_list)
    """
    based_label_idx = get_column_index(adata, based_label)
    to_label_idx = get_column_index(adata, to_label)
    for cell in range(adata.shape[0]):
        if not pd.isna(adata.obs.iat[cell, to_label_idx]):
            region_label = adata.obs.iat[cell, to_label_idx]
            for neighbor in neighbor_list[cell]:
                if pd.isna(adata.obs.iat[neighbor, to_label_idx]):
                    if adata.obs.iat[neighbor, based_label_idx] in based_labels:
                        adata.obs.iat[neighbor, to_label_idx] = region_label
                        
    with plt.rc_context():
        sc.pl.spatial(adata, cmap='magma',color=[to_label],ncols=4, size=1.3,img_key='hires',show=False)
        plt.savefig(f"{save_dir}/label_border.pdf", dpi=80, bbox_inches='tight')
        plt.close()
    return adata



def merge_adjacent_regions(adata,to_label:str,neighbor_list:dict):
    """
    example: region_mapping = merge_adjacent_regions(adata,'Hypo_region',neighbor_list)
    """
    region_mapping = {i: set() for i in adata.obs[to_label].dropna().unique()}
    to_label_idx = get_column_index(adata, to_label)
    for cell in range(adata.shape[0]):
        if not pd.isna(adata.obs.iat[cell, to_label_idx]):
            region_label = adata.obs.iat[cell, to_label_idx]
            for neighbor in neighbor_list[cell]:
                neighbor_label = adata.obs.iat[neighbor, to_label_idx]
                if not pd.isna(neighbor_label) and neighbor_label != region_label:
                    # region_mapping[neighbor_label] = region_label
                    region_mapping[neighbor_label].add(region_label)

    region_mapping = {k: v for k, v in region_mapping.items() if v}
                    
    return region_mapping
    
    
def dict_to_nested_list(input_dict):
    nested_list = []
    for key, values in input_dict.items():
        #       ֵ  ϳ һ   б 
        combined = [key] + list(values)
        #   ӵ Ƕ   б   
        nested_list.append(combined)
    return nested_list
    
    
def merge_and_convert_to_dict(nested_list):
    def merge_lists(list1, list2):
        return sorted(list(set(list1 + list2)))

    merged = True
    while merged:
        merged = False
        for i in range(len(nested_list)):
            for j in range(i + 1, len(nested_list)):
                if set(nested_list[i]) & set(nested_list[j]):
                    nested_list[j] = merge_lists(nested_list[i], nested_list[j])
                    nested_list.pop(i)
                    merged = True
                    break
            if merged:
                break

    result_dict = {}
    for sublist in nested_list:
        key = sublist[0]
        values = set(sublist[1:])
        result_dict[key] = values
    
    return result_dict
    
def rever_dict(dict1):
    reverse_dict = {}
    for key, value in dict1.items():
        if isinstance(value, set):
            for item in value:
                reverse_dict[item] = key
        else:
            reverse_dict[value] = key

    return reverse_dict


def remove_duplicates(input_dict):
    """
    example: 
    region_mapping = remove_duplicates(region_mapping)
    adata.obs['Hypo_region'].replace(region_mapping, inplace=True)
    """
    unique_sets = set()
    result = {}
    
    for key, value in input_dict.items():
        combined_set = frozenset([key]) | frozenset(value if isinstance(value, set) else [value])
        if combined_set not in unique_sets:
            unique_sets.add(combined_set)
            result[key] = value

    return result


def label_peri_one_cells(adata, to_label: str, based_label: str, based_labels: str,neighbor_list:dict,filtered_label:str,add_key:str):
    """
    example: 
    adata = label_peri_one_cells(adata,'Hypo_region','cell_status','not_Hypo',neighbor_list,'Hypo_reg','layer1')
    """
    if filtered_label == 'Hypo_reg':
        replace_str = 'iuoiuoiuioioi'
    else:
        replace_str = filtered_label + "_"
    #   ȡ      
    based_label_idx = get_column_index(adata, based_label)
    to_label_idx = get_column_index(adata, to_label)
    #     ÿ  ϸ         ھ ϸ   ı ǩ
    for cell in range(adata.shape[0]):
        if not pd.isna(adata.obs.iat[cell, to_label_idx]) and adata.obs.iat[cell, to_label_idx].startswith(filtered_label):
            for neighbor in neighbor_list[cell]:
                if pd.isna(adata.obs.iat[neighbor, to_label_idx]):
                    if adata.obs.iat[neighbor, based_label_idx] == based_labels:
                        new_category = add_key + "_" + adata.obs.iat[cell, to_label_idx].replace(replace_str,"")
                        
                        # ȷ             б   
                        if new_category not in adata.obs[to_label].cat.categories:
                            adata.obs[to_label] = adata.obs[to_label].cat.add_categories([new_category])
                        
                        #       ֵ
                        adata.obs.iat[neighbor, to_label_idx] = new_category
                        
    return adata

def find_indirect_connections(adata,to_label:str,neighbor_list:dict):
    """
    Find indirect connections between different Hypo_regions via unlabelled cells.
    """
    to_label_idx = get_column_index(adata, to_label)
    indirect_connections = []

    for cell in range(adata.shape[0]):
        if pd.isna(adata.obs.iat[cell, to_label_idx]):  # Check unlabelled cells
            connected_regions = set()
            for neighbor in neighbor_list[cell]:
                neighbor_label = adata.obs.iat[neighbor, to_label_idx]
                if not pd.isna(neighbor_label):
                    connected_regions.add(neighbor_label)
                    # print(connected_regions)
            if len(connected_regions) > 1:
                indirect_connections.append(list(connected_regions))
    indirect_connections = merge_and_convert_to_dict(indirect_connections)
    indirect_connections = rever_dict(indirect_connections)

    return indirect_connections


def get_df(adata,counts_key:str,save_dir):
    all_celltype = adata.obs[counts_key].dropna()
    cell_types = all_celltype.unique()

    #     һ   յ  DataFrame    洢   
    mean_expression_df = pd.DataFrame(index=adata.var_names, columns=cell_types)
    cell_numbers = []
    #     ÿ  ϸ     ͵    л    ƽ       
    for cell_type in cell_types:
        if not pd.isna(cell_type):
            cells_in_type = adata.obs[counts_key] == cell_type
            mean_expression = np.mean(adata[cells_in_type].X, axis=0)
            cell_numbers.append(adata[cells_in_type].shape[0])
            mean_expression_df[cell_type] = mean_expression

    mean_expression_df.loc['cell_numbers'] = cell_numbers
    mean_expression_df.loc['sample'] = adata.obs['sample'].tolist()[0:len(cell_types)]
    mean_expression_df.to_csv(f"{save_dir}/mean.tsv",sep="\t")
    return mean_expression_df



def get_df_ratio(adata, counts_key: str, save_dir ,method = 'mean'):
    all_celltype = adata.obs[counts_key].dropna()
    cell_types = all_celltype.unique()

    #     һ   յ  DataFrame    洢   
    mean_expression_ratio_df = pd.DataFrame(index=adata.var_names, columns=cell_types)
    cell_numbers = []

    #     ÿ  ϸ     ͵    л    ƽ        
    for cell_type in cell_types:
        if not pd.isna(cell_type):
            cells_in_type = adata.obs[counts_key] == cell_type
            cell_type_data = adata[cells_in_type].X
            #     ÿ  ϸ    ÿ      ı     
            expression_ratio = cell_type_data / cell_type_data.sum(axis=1)[:, None]
            #      ϸ          ϸ  ÿ       ƽ      
            if method == 'mean' :
                mean_expression_ratio = np.mean(expression_ratio, axis=0)
            elif method == 'median' :
                mean_expression_ratio = np.median(expression_ratio, axis=0)
            cell_numbers.append(adata[cells_in_type].shape[0])
            mean_expression_ratio_df[cell_type] = mean_expression_ratio

    mean_expression_ratio_df.loc['cell_numbers'] = cell_numbers
    mean_expression_ratio_df.loc['sample'] = adata.obs['sample'].tolist()[0:len(cell_types)]
    
    # ȷ      Ŀ¼    
    os.makedirs(save_dir, exist_ok=True)
    mean_expression_ratio_df.to_csv(f"{save_dir}/mean_ratio.tsv", sep="\t")
    return mean_expression_ratio_df

def filter_small_categories(adata, column_name, num = 10):
    """
         ִ   С  10      滻ΪNA  

        :
    - adata:        ݵ AnnData    
    - column_name: Ҫ           

        :
    -  ޸ĺ  AnnData    
    """
    if adata.obs[column_name].dtype != 'category':
        adata.obs[column_name] = adata.obs[column_name].astype('category')
    # ͳ  ÿ      Ƶ  
    category_counts = adata.obs[column_name].value_counts()

    #  ҵ    ִ   С  10     
    small_categories = category_counts[category_counts < num].index

    #     Щ    ֵ 滻ΪNA
    adata.obs[column_name] = adata.obs[column_name].apply(lambda x: x if x not in small_categories else np.nan)
    
    return adata





adata_main.obs[adata_main.uns['mod']['factor_names']] = adata_main.obsm['q05_cell_abundance_w_sf']
pattern_to_remove = "-UKF.*"

sample_list = adata_main.obs['sample'].unique()
celltype_list = ['AC_like', 'G_cycling', 'MES_Core', 'MES_Hypo','NPC_like', 'OPC_like']
for smp in sample_list:
    print(f"----------------{smp} started----------------")
    adata = adata_main[adata_main.obs['sample'] == smp]
    adata.uns['spatial'] = {k: v for k, v in adata.uns['spatial'].items() if k == smp}
    save_dir = f"/home/{smp}"
    mkdir(save_dir)
    adata = QC(adata,save_dir)
    adata.obs[celltype_list].to_csv(f"{save_dir}/{smp}_celltype_anno.tsv",sep="\t")
    print(f"{smp} starting")
    if adata.shape[0] < 1000:
        print(adata.shape)
        print(f"----------------{smp} low quality----------------")
        shutil.rmtree(save_dir)
        continue
    adata = trans2cell(adata,celltype_list)
    with plt.rc_context():
        sc.pl.spatial(adata, cmap='magma',color=['MES_Core','MES_Hypo'],ncols=4, size=1.3,img_key='hires',show=False)
        plt.savefig(f'{save_dir}/all_cell_distribution.pdf', dpi=80, bbox_inches='tight')
        plt.close()
    if adata[:, 'MES_Hypo'].X.max() < 4 :
        print(f"----------------{smp} not Hypoxia----------------")
        shutil.rmtree(save_dir)
        continue
    adata = replace_nan_and_check(adata)
    adata = set_region(adata,"Hypo","Hypo",40,save_dir)
    if (adata.obs['cell_type'] == 'Hypo').sum() < 100:
        print(adata.shape)
        print(f"----------------{smp} without enough Hypo spots----------------")
        shutil.rmtree(save_dir)
        continue
    adata,neighbors = define_neighbors(adata,'Hypo','not_Hypo',save_dir)
    adata,neighbor_list = label_hypo_regions(adata, neighbors, 'cell_status', ['core', 'peri_core'], 'Hypo_region')
    adata = label_border_cells(adata,'Hypo_region','cell_status',['border'],neighbor_list,save_dir)
    adata = label_border_cells(adata,'Hypo_region','cell_status',['border','isolated'],neighbor_list,save_dir)
    region_mapping = merge_adjacent_regions(adata,'Hypo_region',neighbor_list)
    region_mapping = dict_to_nested_list(region_mapping)
    region_mapping = merge_and_convert_to_dict(region_mapping)
    region_mapping = rever_dict(region_mapping)
    adata.obs['Hypo_region'].replace(region_mapping, inplace=True)
    print(adata.obs['Hypo_region'].dtype)
    print(adata.obs['Hypo_region'])
    adata = filter_small_categories(adata,'Hypo_region',20)
    #indirect_connections = find_indirect_connections(adata,'Hypo_region',neighbor_list)
    #adata.obs['Hypo_region'].replace(indirect_connections, inplace=True)
    print(adata.obs['Hypo_region'].dtype)
    print(adata.obs['Hypo_region'])
    #adata.obs['Hypo_region'] = adata.obs['Hypo_region'].astype('category')
    sc.pl.spatial(adata,color='Hypo_region',show=False)
    adata = label_peri_one_cells(adata,'Hypo_region','cell_status','not_Hypo',neighbor_list,'Hypo_reg','layer1')
    adata = label_peri_one_cells(adata,'Hypo_region','cell_status','not_Hypo',neighbor_list,'layer1','layer2')
    adata = label_peri_one_cells(adata,'Hypo_region','cell_status','not_Hypo',neighbor_list,'layer2','layer3')
    adata = label_peri_one_cells(adata,'Hypo_region','cell_status','not_Hypo',neighbor_list,'layer3','layer4')
    adata = label_peri_one_cells(adata,'Hypo_region','cell_status','not_Hypo',neighbor_list,'layer4','layer5')
    adata = label_peri_one_cells(adata,'Hypo_region','cell_status','not_Hypo',neighbor_list,'layer5','layer6')
    with plt.rc_context():
        sc.pl.spatial(adata, cmap='magma',color=['Hypo_region'],ncols=4, size=1.3,img_key='hires',show=False)
        plt.savefig(f"{save_dir}/Hypo_region_final.pdf", dpi=80, bbox_inches='tight')
        plt.close()
    mean_df = get_df(adata,'Hypo_region',save_dir)
    mean_ratio_df = get_df_ratio(adata,'Hypo_region',save_dir,method='mean')
    adata.obs_names = adata.obs_names.str.replace(pattern_to_remove, "", regex=True)
    adata.write(f"{save_dir}/{smp}_border_anno.h5ad")
    adata.obs[['cell_type', 'cell_status', 'Hypo_region']].to_csv(f"{save_dir}/{smp}_region_anno.tsv",sep="\t")
    print(f"----------------{smp} finished----------------")


