from __future__ import annotations

import logging
from time import time

import numpy as np
import pandas as pd
from sqlalchemy import inspect, select, exc
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ..utils.sql import Base, Designs, Poses, Residues
from ..resources.job import job_resources_factory

# Globals
logger = logging.getLogger(__name__)
# Session = sessionmaker()


def insert_dataframe(session: Session, _table: Base, df: pd.DataFrame):  # -> list[int]:
    """Take a MultiIndex(pose, design, index) Residues DataFrame and insert/update values into the Residues SQL DataBase

    Returns:
        The id of each of the inserted entries in the Database
    """
    # table = Residues
    # job = job_resources_factory()
    # with job.db.session() as session:
    insert_stmt = insert(_table)
    # Get the columns that should be updated
    new_columns = df.index.names + df.columns.to_list()
    # logger.debug(f'Provided columns: {new_columns}')
    excluded_columns = insert_stmt.excluded
    update_columns = [c for c in excluded_columns if c.name in new_columns]
    update_dict = {getattr(c, 'name'): c for c in update_columns if not c.primary_key}
    table = _table.__table__
    # Find relevant column indicators to parse the non-primary key non-nullable columns
    primary_keys = [key for key in table.primary_key]
    non_null_keys = [col for col in table.columns if not col.nullable]
    index_keys = [key for key in non_null_keys if key not in primary_keys]

    # do_update_stmt = insert_stmt.on_conflict_do_update(
    #     index_elements=index_keys,  # primary_keys,
    #     set_=update_dict
    # )
    # # Can't insert with .returning() until version 2.0...
    # # try:
    # #     result = session.execute(do_update_stmt.returning(table.id), df.reset_index().to_dict('records'))
    # # except exc.CompileError as error:
    # #     logger.error(error)
    # #     try:
    # #         result = session.execute(insert_stmt.returning(table.id), df.reset_index().to_dict('records'))
    # #     except exc.CompileError as _error:
    # #         logger.error(_error)
    # # try:
    # This works using insert with conflict, however, doesn't return the auto-incremented ids
    # result = session.execute(do_update_stmt, df.reset_index().to_dict('records'))
    # result = session.execute(insert_stmt, df.reset_index().to_dict('records'))
    start_time = time()
    session.execute(insert_stmt, df.reset_index().to_dict('records'))
    logger.info(f'Transaction took {time() - start_time:8f}s')

    session.commit()
    # input('is this deleting?')
    # foreign_key = [key.column.name for key in table.foreign_keys]
    for key in table.foreign_keys:
        foreign_key_name = key.column.name
        table2 = key.column.table
        # Repeat the Find procedure for table2
        # Find relevant column indicators to parse the non-primary key non-nullable columns
        primary_keys2 = [key for key in table2.primary_key]
        non_null_keys2 = [col for col in table2.columns if not col.nullable]
        index_keys2 = [key for key in non_null_keys2 if key not in primary_keys2]
        # Todo this statement fails due to the error:
        #  This backend (sqlite) does not support multiple-table criteria within UPDATE
        #  This doesn't appear to be a multiple-table update, but a multiple-table criteria,
        #  which is supported by sqlite...
        # foreign_key_update_stmt = table.update()\
        #     .values({key.parent.name: key.column})\
        #     .where(*tuple(key1 == key2 for key1, key2 in zip(index_keys, index_keys2)))
        # logger.info(foreign_key_update_stmt)

        select_stmt = select(key.column).where(key.parent.is_(None))\
            .where(*tuple(key1 == key2 for key1, key2 in zip(index_keys, index_keys2))).scalar_subquery()
        foreign_key_update_stmt2 = table.update()\
            .values({key.parent.name: select_stmt})
        logger.info(foreign_key_update_stmt2)
        # session.execute(foreign_key_update_stmt)
        start_time = time()
        session.execute(foreign_key_update_stmt2)
        logger.info(f'Transaction took {time() - start_time:8f}s')

    session.commit()
    # return result

    # # ORM based method which updates objects with each row .id (Takes much longer time
    # # - https://benchling.engineering/sqlalchemy-batch-inserts-a-module-for-when-youre-inserting-thousands-of-rows-and-it-s-slow-16ece0ef5bf7
    # new_objects = [table(**record) for record in df.reset_index().to_dict('records')]
    # # select(table)
    # session.add_all(new_objects)
    # session.commit()
    # # result = [_object.id for _object in new_objects]
    # return [_object.id for _object in new_objects]

# - DRAFTING SPACE -
# For updating foreign keys, I can use the ORM method, which seems a bit simpler if I refactor properly
# table_instance = table(**record)
# foreign_instance = table1(**other_record)
# table_instance.append(foreign_instance)
# Or using UPDATE table foreign_key_id WHERE table.non_nullable_column == foreign_key_table.non_nullable_column
#  table = table.__table__
#  primary_keys = [key.name for key in table.primary_key]
#  non_null_keys = [col.name for col in table.columns if not col.nullable]
#  index_keys = [key for key in non_null_keys if key not in primary_keys]
# for key in table.foreign_keys:
#  foreign_key_name = key.column.name
#  table2 = key.column.table
#  primary_keys2 = [key.name for key in table2.primary_key]
#  non_null_keys2 = [col.name for col in table2.columns if not col.nullable]
#  index_keys2 = [key for key in non_null_keys2 if key not in primary_keys2]
# table.update()
#      .values({key.parent.name: key.column})
#      .where(*tuple(key1 == key2 for key1, key2 in zip(index_keys1, index_keys2)))


def upsert_dataframe(session: Session, _table: Base, df: pd.DataFrame) -> list[int]:
    """Take a MultiIndex(pose, design, index) Residues DataFrame and insert/update values into the Residues SQL DataBase

    This dataframe must have a column 'id' in order to upsert. This should be retrieved from the db
    """
    # table = Residues
    # job = job_resources_factory()
    # with job.db.session() as session:
    insert_stmt = insert(_table)
    # Get the columns that should be updated
    new_columns = df.index.names + df.columns.to_list()
    # logger.debug(f'Provided columns: {new_columns}')
    excluded_columns = insert_stmt.excluded
    update_columns = [c for c in excluded_columns if c.name in new_columns]
    update_dict = {getattr(c, 'name'): c for c in update_columns if not c.primary_key}
    table = _table.__table__
    # Find relevant column indicators to parse the non-primary key non-nullable columns
    primary_keys = [key.name for key in table.primary_key]
    non_null_keys = [col.name for col in table.columns if not col.nullable]
    index_keys = [key for key in non_null_keys if key not in primary_keys]

    do_update_stmt = insert_stmt.on_conflict_do_update(
        index_elements=index_keys,  # primary_keys,
        set_=update_dict
    )

    result = session.execute(do_update_stmt, df.reset_index().set_index('id').to_dict('records'))
    session.commit()

    return result.scalars().all()


def format_residues_df_for_write(df: pd.DataFrame) -> pd.DataFrame:
    """Take a typical per-residue DataFrame and orient the top column level (level=0) containing the residue numbers on
    the index innermost level

    Args:
        df: A per-residue DataFrame to transform
    Returns:
        The transformed dataframe
    """
    # df.sort_index(inplace=True)
    df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
    # residue_metric_columns = residues.columns.levels[-1].to_list()
    # self.log.debug(f'Residues metrics present: {residue_metric_columns}')

    # Add the pose identifier to the dataframe
    # df = pd.concat([df], keys=[str(self)], axis=0)
    # Place the residue indices from the column names into the index at position -1
    df = df.stack(0)
    # df.index.set_names(['pose', 'design', 'index'], inplace=True)
    df.index.set_names('index', level=-1, inplace=True)

    return df


def write_dataframe(designs: pd.DataFrame = None, residues: pd.DataFrame = None, poses: pd.DataFrame = None,
                    update: bool = False):
    """Format each possible DataFrame type for output via csv or SQL database

    Args:
        designs: The typical per-design metric DataFrame where each index is the design id and the columns are
            design metrics
        residues: The typical per-residue metric DataFrame where each index is the design id and the columns are
            (residue index, residue metric)
        poses: The typical per-pose metric DataFrame where each index is the pose id and the columns are
            pose metrics
        update: Whether the output identifiers are already present in the metrics
    """
    job = job_resources_factory()
    # engine = job.db.engine
    if update:
        dataframe_function = upsert_dataframe
    else:
        dataframe_function = insert_dataframe

    with job.db.session() as session:
        if poses is not None:
            poses.replace({np.nan: None}, inplace=True)
            result = dataframe_function(session, _table=Poses, df=poses)
            table = Poses.__tablename__
            # poses.to_sql(table, con=engine, if_exists='append', index=True)
            # #              dtype=sql.Base.metadata.table[table])
            logger.info(f'Wrote {table} metrics to DataBase {job.internal_db}')

            return result

        if designs is not None:
            designs.replace({np.nan: None}, inplace=True)
            result = dataframe_function(session, _table=Designs, df=designs)
            table = Designs.__tablename__
            # designs.to_sql(table, con=engine, if_exists='append', index=True)
            # #                dtype=sql.Base.metadata.table[table])
            logger.info(f'Wrote {table} metrics to DataBase {job.internal_db}')

            return result

        if residues is not None:
            residues = format_residues_df_for_write(residues)
            residues.replace({np.nan: None}, inplace=True)
            result = dataframe_function(session, _table=Residues, df=residues)
            table = Residues.__tablename__
            # residues.to_sql(table, con=engine, index=True)  # if_exists='append',
            # # Todo Ensure that the 'id' column is present
            # stmt = select(Designs).where(Designs.pose.in_(pose_ids))  # text('''SELECT * from residues''')
            # with job.db.session() as session:
            #     next_available_id = Designs.next_key(session)
            #     stmt = select(Designs).order_by(Designs.id.desc()).limit(1)
            #     session.scalars(stmt).first().id
            #     results = session.scalars(stmt)  # execute(stmt)
            #     upsert_dataframe(table, residues)
            logger.info(f'Wrote {table} metrics to DataBase {job.internal_db}')

            return result
