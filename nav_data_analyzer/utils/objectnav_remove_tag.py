import sys
import argparse

from numpy import integer, short
from sqlalchemy import create_engine, Column, String, Float, Integer, ForeignKey, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

#os.chdir('/mnt/sfs-turbo-workflow/data-platform/')

# ORM 基类
Base = declarative_base()

# 表模型定义
class TaskDatasetObjectNav(Base):
    __tablename__ = "task_datasets_objectnav"
    __table_args__ = {'schema': 'public'}

    id = Column(String(36), primary_key=True)
    scene_type = Column(String(50), nullable=False)
    split = Column(String(10), nullable=False)
    scene_name = Column(String(255), nullable=False)
    object_category = Column(String(255), nullable=False)
    geodesic_distance = Column(Float, nullable=False)
    euclidean_distance = Column(Float, nullable=False)
    nav_complexity_ratio = Column(Float, nullable=False)
    recipe_tags = Column(ARRAY(String), default=[])

class TrajDatasetObjectNav(Base):
    __tablename__ = "traj_datasets_objectnav"
    __table_args__ = {'schema': 'public'}

    id = Column(String(36), primary_key=True)                     # 唯一轨迹ID
    gen_traj_method = Column(String(50), nullable=False)          # 轨迹生成方法
    task_id = Column(String(36), nullable=False)  # 关联任务ID
    success = Column(Integer, nullable=False)                     # 是否成功完成任务
    spl = Column(Float)                                           # SPL指标
    traj_len = Column(Integer)                                      # 轨迹长度
    experiment_name = Column(String(255))                         # 实验名称
    # 添加一个recipe_tags字段，类型为ARRAY(String)，默认值为空列表
    recipe_tags = Column(ARRAY(String), default=[])

def precheck_args(args):
    invalid_args = []
    if args.dataset_type not in ["task_datasets", "traj_datasets"]:
        print(f"❌ 参数 dataset_type {args.dataset_type} 必须是 'task_datasets' 或 'traj_datasets'")
        invalid_args.append("dataset_type")
    return invalid_args

# 主程序逻辑
def main(args):
    encoded_password = quote_plus(args.db_password)
    db_url = f"postgresql+psycopg2://{args.db_user}:{encoded_password}@{args.db_host}:{args.db_port}/{args.db_name}"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    print(f"🔗 连接数据库：{db_url}")

    # 如果是task_datasets，就查询并操作 TaskDatasetObjectNav 表
    # 如果是traj_datasets，就查询并操作 TrajDatasetObjectNav 表
    # 遍历存在指定标签的数据记录，删除该标签
    try:
        if args.dataset_type == "task_datasets":
            records = session.query(TaskDatasetObjectNav).filter(
                TaskDatasetObjectNav.recipe_tags.any(args.remove_recipe_tag)
            ).all()
        elif args.dataset_type == "traj_datasets":
            records = session.query(TrajDatasetObjectNav).filter(
                TrajDatasetObjectNav.recipe_tags.any(args.remove_recipe_tag)
            ).all()

        print(f"🗂️ 找到 {len(records)} 条包含标签 '{args.remove_recipe_tag}' 的记录，准备删除该标签...")

        for record in records:
            if args.dataset_type == "task_datasets":
                record.recipe_tags.remove(args.remove_recipe_tag)
            elif args.dataset_type == "traj_datasets":
                record.recipe_tags.remove(args.remove_recipe_tag)

            session.add(record)

        session.commit()
        print(f"✅ 成功删除标签 '{args.remove_recipe_tag}'，共处理 {len(records)} 条记录。")

    except SQLAlchemyError as e:
        session.rollback()
        print(f"❌ 数据库操作失败: {e}")
    finally:
        session.close()
        print("🔒 数据库连接已关闭。")

# argparse 参数解析
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除数据库中指定数据集记录的标签")
    parser.add_argument("--dataset_type", default="traj_datasets", help="task_datasets or traj_datasets")
    parser.add_argument("--remove_recipe_tag", default='hm3d_v1_hd_l3mvn_refine_v2_30k', help="数据配方标签")

    # database connection parameters
    parser.add_argument("--db_user", default='dbadmin', help="数据库用户名")
    parser.add_argument("--db_password", default='xxx', help="数据库密码")
    parser.add_argument("--db_host", default="dws-z00562901.dws.myhuaweiclouds.com", help="数据库主机")
    parser.add_argument("--db_port", default="8000", help="数据库端口")
    parser.add_argument("--db_name", default='postgres' , help="数据库名称")

    args = parser.parse_args()
    invalid_args = precheck_args(args)
    if len(invalid_args) == 0:
        for arg, value in vars(args).items():
            print(f"🔧 参数 {arg}: {value}")
        main(args)
    elif len(invalid_args) > 0:
        for arg in invalid_args:
            print(f"❌ 参数 {arg} 无效，请检查后重新运行脚本。")
        sys.exit(1)