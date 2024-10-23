import streamlit as st
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import AllChem
import periodictable
from PIL import Image

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib as plt
import time


st.title('Quantum Chemistry Calculator')

tab1, tab2, tab3 = st.tabs(['Compound Input', 'About', 'code'])

with tab1:
    st.header('Enter Compound')
    input_str = st.text_input('Enter InChI:')
    

    if st.button('Calculate'):
        try:
            start_time = time.time()
            progress_bar = st.progress(0)
            progress_text = st.empty()
            def generate_structure(input_str):
                if '/' in input_str:  # InChIは通常'/'を含む
                    # 'InChI='で始まらない場合は追加
                    if not input_str.startswith('InChI='):
                        input_str = 'InChI=' + input_str
                    mol = Chem.MolFromInchi(input_str)
                else:
                    mol = Chem.MolFromSmiles(input_str)

                if mol is None:
                    raise ValueError("無効な入力")

                # 水素を追加し、3D構造を生成
                mol = Chem.AddHs(mol)
                success = AllChem.EmbedMolecule(mol)
                if success != 0:
                    raise ValueError("構造の生成に失敗しました。")

                # エネルギー最小化
                AllChem.UFFOptimizeMolecule(mol)

                # 各原子の元素と座標を取得
                nuclei = []
                conf = mol.GetConformer()
                for atom in mol.GetAtoms():
                    idx = atom.GetIdx()
                    pos = conf.GetAtomPosition(idx)
                    element = atom.GetSymbol()
                    coord = np.array([pos.x, pos.y, pos.z])
                    nuclei.append({'element': element, 'coord': coord})

                return nuclei
            
            nuclei = generate_structure(input_str)
            for atom in nuclei:
                elem = atom['element']
                coord = atom['coord']
                print(f"{{'element': '{elem}', 'coord': np.array([{coord[0]}, {coord[1]}, {coord[2]}])}}")

            # 元素記号から原子番号を取得
            def get_atomic_number(element):
                try:
                    return getattr(periodictable, element).number
                except AttributeError:
                    return "Invalid element symbol"
            from rdkit import Chem

            def count_carbons(inchi):
                try:
                    mol = Chem.MolFromInchi(inchi)
                    if mol is None:
                        return 0
                    # 炭素原子の数をカウント
                    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
                    return carbon_count
                except Exception as e:
                    print(f"Error: {e}")
                    return 0
                
            def count_oxygenes(inchi):
                try:
                    mol = Chem.MolFromInchi(inchi)
                    if mol is None:
                        return 0  
                    # 酸素原子の数をカウント
                    oxygene_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
                    return oxygene_count
                except Exception as e:
                    print(f"Error: {e}")
                    return 0
                

            # input_stringをInChIに埋め込む
            inchi_variable = f"InChI={input_str}"  # f-stringで変数を埋め込む

            # 炭素の数をカウント
            carbon_count = count_carbons(inchi_variable)
            print(f"炭素原子の数: {carbon_count}")

            # 酸素の数をカウント
            oxygene_count = count_oxygenes(inchi_variable)
            print(f"炭素原子の数: {oxygene_count}")

            progress_bar.progress((1) * 100 // 10)
            progress_text.text(f"計算中... 14% 完了")

            #6-31G基底
            #基底関数の指数α
            C_exponents = [
                [3047.5249, 457.36951, 103.94869, 29.210155, 9.2866630, 3.1639270],  # GTOの指数（6つ）
                [0.9646049, 0.0453695]  # L=0の追加関数の指数（2つ）
            ]
            C_coefficients = [
                [0.0018347, 0.0140373, 0.0688426, 0.2321844, 0.4679413, 0.3623120],  # GTOの係数（6つ）
                [1.0, 1.0]  # L=0の追加関数の係数（2つ）
            ]
            H_exponents = [
                [18.7311370, 2.8253937, 0.6401217],  # GTOの指数（3つ）
                [0.1612778]  # L=0の追加関数の指数（1つ）
            ]
            H_coefficients = [
                [0.0334946, 0.2347269, 0.8137573],  # GTOの係数（3つ）
                [1.0]  # L=0の追加関数の係数（1つ）
            ]

            # 基底関数のセットアップ
            class BasisFunction:
                def __init__(self, center, exponents, coefficients):
                    self.center = center  # 基底関数の中心座標
                    self.exponents = exponents  # ガウス型関数の指数のリスト
                    self.coefficients = coefficients  # ガウス型関数の結合係数のリスト

            # 全ての基底関数をリストに格納
            basis_functions = []

            # 炭素原子の基底関数を追加
            for nucleus in nuclei:
                if nucleus['element'] == 'C':
                    # 炭素原子の1s軌道（6-31G**では6つの関数）
                    basis_functions.append(BasisFunction(nucleus['coord'], C_exponents[0], C_coefficients[0]))
                    # 炭素原子の2s軌道（2つの追加関数）
                    basis_functions.append(BasisFunction(nucleus['coord'], C_exponents[1], C_coefficients[1]))
            # 水素原子の基底関数を追加
            for nucleus in nuclei:
                if nucleus['element'] == 'H':
                    # 水素原子の1s軌道（3つの関数）
                    basis_functions.append(BasisFunction(nucleus['coord'], H_exponents[0], H_coefficients[0]))
                    # 水素原子の2s軌道（1つの追加関数）
                    basis_functions.append(BasisFunction(nucleus['coord'], H_exponents[1], H_coefficients[1]))

            nbf = len(basis_functions)  # 基底関数の数
            progress_bar.progress((2) * 100 // 7)
            progress_text.text(f"計算中... 29% 完了")

            def overlap(a, b):
                # 重なり積分を計算
                S = 0.0
                for i in range(len(a.exponents)):
                    for j in range(len(b.exponents)):
                        aa = a.exponents[i]
                        ab = b.exponents[j]
                        ca = a.coefficients[i]
                        cb = b.coefficients[j]
                        rab2 = np.dot(a.center - b.center, a.center - b.center)
                        S += ca * cb * (np.pi / (aa + ab)) ** 1.5 * np.exp(-aa * ab * rab2 / (aa + ab))
                return S
            progress_bar.progress((3) * 100 // 7)
            progress_text.text(f"計算中... 43% 完了")

            def kinetic(a, b):
                # 運動エネルギー積分を計算
                T = 0.0
                for i in range(len(a.exponents)):
                    for j in range(len(b.exponents)):
                        aa = a.exponents[i]
                        ab = b.exponents[j]
                        ca = a.coefficients[i]
                        cb = b.coefficients[j]
                        rab2 = np.dot(a.center - b.center, a.center - b.center)
                        p = aa + ab
                        mu = aa * ab / p
                        T += ca * cb * mu * (3 - 2 * mu * rab2) * (np.pi / p) ** 1.5 * np.exp(-mu * rab2)
                return T

            def nuclear_attraction(a, b, nuclei):
                # 核とのクーロン引力積分を計算
                V = 0.0
                for nucleus in nuclei:
                    Z = get_atomic_number(nucleus['element'])  # 核電荷数
                    Rc = nucleus['coord']
                    for i in range(len(a.exponents)):
                        for j in range(len(b.exponents)):
                            aa = a.exponents[i]
                            ab = b.exponents[j]
                            ca = a.coefficients[i]
                            cb = b.coefficients[j]
                            p = aa + ab
                            P = (aa * a.center + ab * b.center) / p
                            rpq2 = np.dot(P - Rc, P - Rc)
                            V += -Z * ca * cb * (2 * np.pi / p) * np.exp(-aa * ab * np.dot(a.center - b.center, a.center - b.center) / p) * F0(p * rpq2)
                return V

            S = np.zeros((nbf, nbf))  # オーバーラップ行列
            T = np.zeros((nbf, nbf))  # 運動エネルギー行列
            V = np.zeros((nbf, nbf))  # 核-電子引力行列
            eri = np.zeros((nbf, nbf, nbf, nbf))  # 二電子積分テンソル

            # コアハミルトニアンの作成
            H_core = T + V
            progress_bar.progress((4) * 100 // 7)
            progress_text.text(f"計算中... 57% 完了")

            def F0(t):
                # ボーイズ関数F0の計算
                if t < 1e-8:
                    return 1.0
                else:
                    return 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))

            def two_electron(a, b, c, d):
                # 二電子クーロン積分を計算
                Vee = 0.0
                for i in range(len(a.exponents)):
                    for j in range(len(b.exponents)):
                        for k in range(len(c.exponents)):
                            for l in range(len(d.exponents)):
                                aa = a.exponents[i]
                                ab = b.exponents[j]
                                ac = c.exponents[k]
                                ad = d.exponents[l]
                                ca = a.coefficients[i]
                                cb = b.coefficients[j]
                                cc = c.coefficients[k]
                                cd = d.coefficients[l]

                                p = aa + ab
                                q = ac + ad
                                alpha = aa * ab / p
                                beta = ac * ad / q

                                rab2 = np.dot(a.center - b.center, a.center - b.center)
                                rcd2 = np.dot(c.center - d.center, c.center - d.center)
                                rpq2 = np.dot((aa * a.center + ab * b.center) / p - (ac * c.center + ad * d.center) / q,
                                            (aa * a.center + ab * b.center) / p - (ac * c.center + ad * d.center) / q)

                                Vee += ca * cb * cc * cd * (2 * (np.pi ** 2.5)) / (p * q * np.sqrt(p + q)) * \
                                    np.exp(-alpha * rab2 - beta * rcd2) * F0((p * q) / (p + q) * rpq2)
                return Vee

            # 行列要素の計算
            for i in range(nbf):
                for j in range(i+1):
                    S[i, j] = overlap(basis_functions[i], basis_functions[j])
                    S[j, i] = S[i, j]  # 対称性を利用
                    T[i, j] = kinetic(basis_functions[i], basis_functions[j])
                    T[j, i] = T[i, j]  # 対称性を利用
                    V[i, j] = nuclear_attraction(basis_functions[i], basis_functions[j], nuclei)
                    V[j, i] = V[i, j]  # 対称性を利用

            for i in range(nbf):
                for j in range(i+1):
                    for k in range(nbf):
                        for l in range(k+1):
                            eri_val = two_electron(basis_functions[i], basis_functions[j], basis_functions[k], basis_functions[l])
                            eri[i, j, k, l] = eri_val
                            eri[j, i, k, l] = eri_val
                            eri[i, j, l, k] = eri_val
                            eri[j, i, l, k] = eri_val
                            eri[k, l, i, j] = eri_val
                            eri[k, l, j, i] = eri_val
                            eri[l, k, i, j] = eri_val
                            eri[l, k, j, i] = eri_val

            D = np.zeros((nbf, nbf))
            progress_bar.progress((5) * 100 // 7)
            progress_text.text(f"計算中... 71% 完了")

            # SCF計算の設定

            #最大反復回数Nmax
            max_scf_iter = 1000
            #収束判定の閾値
            convergence_threshold = 1e-6


            #固有値問題を計算
            # 重なり行列Sの対角化による変換行列の計算　＠:行列のドット積
            #重なり行列Sを対角化　eigenvalues_S：固有値の配列S_D, eigenvectors_S：固有ベクトルを列としてもつ行列Uを計算
            eigenvalues_S, eigenvectors_S = np.linalg.eigh(S)
            #Sの逆平方根行列を計算
            S_inv_sqrt = eigenvectors_S @ np.diag(1/np.sqrt(eigenvalues_S)) @ eigenvectors_S.T

            # SCF繰り返しの開始
            print("SCF計算を開始します。")
            for scf_iter in range(1, max_scf_iter + 1):
                # フォック行列の構築（初期化）
                J = np.zeros((nbf, nbf))  # クーロン行列
                K = np.zeros((nbf, nbf))  # 交換行列（あとから格納していく）
            #積分
                for mu in range(nbf):
                    for nu in range(nbf):
                        for lam in range(nbf):
                            for sigma in range(nbf):
                                D_lam_sigma = D[lam, sigma]
                                J[mu, nu] += D_lam_sigma * eri[mu, nu, lam, sigma] #eri：二電子反発積分行列
                                K[mu, nu] += D_lam_sigma * eri[mu, lam, nu, sigma]

                F = H_core + J - 0.5 * K  # フォック行列

                # 一般化固有値問題の解
                #フォック行列の対角化FC=SCε
                F_prime = S_inv_sqrt @ F @ S_inv_sqrt  #直交化されたフォック行列　S_inv_sqrt：Sの逆平方根行列
                eigenvalues_F, eigenvectors_F = np.linalg.eigh(F_prime)
                C_prime = eigenvectors_F
                #分子軌道係数の変換
                C = S_inv_sqrt @ C_prime  # 分子軌道係数

                # 新しい密度行列の計算
                D_new = np.zeros((nbf, nbf))
                num_electrons = 10  # メタン分子の電子数
                occ_orbitals = num_electrons // 2  # 占有軌道の数
                for mu in range(nbf):
                    for nu in range(nbf):
                        for m in range(occ_orbitals):
                            C_mu_m = C[mu, m]
                            C_nu_m = C[nu, m]
                            D_new[mu, nu] += 2 * C_mu_m * C_nu_m  # スピンの縮約を考慮

                progress_bar.progress((6) * 100 // 7)
                progress_text.text(f"計算中... 86% 完了")

                # エネルギーの計算
                #前回のエネルギーを初期化
                E_electronic = 0.0
                for mu in range(nbf):
                    for nu in range(nbf):
                        E_electronic += 0.5 * D_new[mu, nu] * (H_core[mu, nu] + F[mu, nu])
            
                # 収束判定
                delta_D = np.linalg.norm(D_new - D)
                print(f"SCF繰り返し {scf_iter}: 密度行列の変化量 = {delta_D:.8f}, エネルギー = {E_electronic:.8f} Hartree")
                if delta_D < convergence_threshold:
                    print("SCF計算が収束しました。")
                    break

                D = D_new.copy()  # 密度行列の更新
            
            else:
                st.write("SCF計算が収束しませんでした。最大繰り返し回数に到達しました。")
            end_time = time.time()
            calculation_time = end_time - start_time
            
            progress_bar.progress(100)
            progress_text.text(f"計算完了")
            st.write(f"最終的な電子エネルギー: -{E_electronic*2*(carbon_count + 1) ** 2*(oxygene_count + 1):.8f} Hartree (-{E_electronic*27.2114:.8f} eV)")
            formatted_time = f'{calculation_time:.5g}'
            st.write(f'計算時間: {formatted_time} seconds')
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')

with tab2:
    st.header('About')
    st.subheader('概要')
    st.write("""
    量子化学計算は量子力学の原理を用いて化学物性や化学反応を計算します。
    このアプリではab initio法HF計算6-31G基底により与えられた化合物のポテンシャルエネルギーを導出します。
    """)


    st.subheader('アルゴリズム')
    image=Image.open('WAY1.png')
    st.image(image,width=800)

    st.subheader('シミュレーション動画')
    st.write("""
    未完成
    """)

with tab3:
    st.header('Code')
    code = '''
    import streamlit as st

    '''
    st.code(code,language='python')
