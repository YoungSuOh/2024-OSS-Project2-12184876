import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ratings.dat 파일 로드
column_names = ['userId', 'movieId', 'rating', 'timestamp']  # 컬럼 이름 지정 -> movielens의 README.txt를 참고하여 진행
ratings = pd.read_csv('ratings.dat', sep='::', names=column_names, engine='python')  # ratings.dat 파일을 읽어와서 DataFrame으로 변환

# User x Item 행렬 생성
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)  # User x Item 행렬 생성, NaN 값은 0으로 채움


num_users, num_items = 6040, 3952 # 행렬 크기를 6040 x 3952로 맞춤
user_item_matrix = user_item_matrix.reindex(index=range(1, num_users + 1), columns=range(1, num_items + 1), fill_value=0)  # 행렬 크기를 고정된 크기로 맞추고 빈 값은 0으로 채움

# K-Means를 사용하여 3개의 그룹으로 사용자 분류
kmeans = KMeans(n_clusters=3, random_state=42)  # KMeans 객체를 생성, 클러스터 수는 3으로 설정
kmeans.fit(user_item_matrix)  # KMeans 클러스터링을 User x Item 행렬에 적용
labels = kmeans.labels_  # 각 사용자의 클러스터 레이블을 얻음

# 그룹 인덱스 가져오기
groups = [np.where(labels == i)[0] for i in range(3)]  # 각 클러스터에 속하는 사용자 인덱스를 얻음

# 각 그룹에 대한 상위 10개 추천 결과 얻기 위한 알고리즘들 정의

# Additive Utilitarian (AU) - 그룹의 모든 사용자 평점을 합산하여 아이템별 점수를 계산
def au(group_ratings): 
    return group_ratings.sum(axis=0)

# Average (Avg) - 그룹의 모든 사용자 평점의 평균을 계산
def avg(group_ratings): 
    return group_ratings.mean(axis=0)

# Simple Count (SC) - 그룹의 각 아이템에 평점을 매긴 사용자 수를 계산
def sc(group_ratings):
    return (group_ratings > 0).sum(axis=0)

# Approval Voting (AV) - 그룹의 각 아이템에 대해 4점 이상의 평점을 매긴 사용자 수를 계산
def av(group_ratings, threshold=4):
    return (group_ratings >= threshold).sum(axis=0)

# Borda Count (BC) - 사용자들의 아이템에 대한 랭킹 점수의 평균을 계산
def bc(group_ratings):
    ranks = np.argsort(np.argsort(-group_ratings, axis=1), axis=1)  # 각 사용자의 평점에 대한 랭킹을 계산
    return ranks.mean(axis=0)  # 랭킹의 평균을 계산

# Copeland Rule (CR) - 각 아이템을 다른 모든 아이템과 비교하여 승리 횟수를 계산
def cr(group_ratings):
    win_count = np.zeros(group_ratings.shape[1])  # 승리 횟수를 저장할 배열 초기화
    for i in range(group_ratings.shape[1]):  # 각 아이템 i에 대해
        for j in range(i+1, group_ratings.shape[1]):  # 다른 아이템 j와 비교
            win_count[i] += (group_ratings[:, i] > group_ratings[:, j]).sum()  # i의 평점이 j보다 높은 사용자의 수를 더함
            win_count[j] += (group_ratings[:, j] > group_ratings[:, i]).sum()  # j의 평점이 i보다 높은 사용자의 수를 더함
    return win_count

# 각 알고리즘을 각 그룹에 적용하여 상위 10개 추천 결과 얻기
algorithms = [au, avg, sc, av, bc, cr]  # 사용할 알고리즘 목록
algorithm_names = ['AU', 'Avg', 'SC', 'AV', 'BC', 'CR']  # 알고리즘 이름 목록

top_10_recommendations = {}  # 최종 추천 결과를 저장할 dictionary

for i, group in enumerate(groups):  # 각 그룹에 대해
    group_ratings = user_item_matrix.iloc[group]  # 해당 그룹의 사용자 평점 행렬을 가져옴
    top_10_recommendations[f'Group {i+1}'] = {}  # 그룹 이름으로 딕셔너리 항목 생성
    for algo, name in zip(algorithms, algorithm_names):  # 각 알고리즘에 대해
        if name == 'AV':  # Approval Voting 알고리즘의 경우
            scores = algo(group_ratings.values, threshold=4)  # 문제에서 주어진 4점 이상 평점
        else:
            scores = algo(group_ratings.values)  # 다른 알고리즘 호출
        top_10_items = np.argsort(scores)[-10:][::-1] + 1  # 상위 10개 아이템의 인덱스를 가져옴, 인덱스를 영화 ID로 변환하기 위해 1을 더함
        top_10_recommendations[f'Group {i+1}'][name] = top_10_items  # 결과를 딕셔너리에 저장

# 결과 출력
for group, results in top_10_recommendations.items():  # 각 그룹에 대해
    print(f"\n{group} Recommendations:")  # 그룹 이름 출력
    for algo, items in results.items():  # 각 알고리즘에 대해
        print(f"{algo}: {items}")  # 알고리즘 이름과 추천 아이템 출력

# 결과를 데이터프레임으로 변환
results_df = pd.DataFrame.from_dict(top_10_recommendations, orient='index')  # 추천 결과를 데이터프레임으로 변환

# 실행 결과를 csv 파일로 저장
results_df.to_csv('result.csv')