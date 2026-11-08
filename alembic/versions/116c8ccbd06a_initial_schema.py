"""Initial schema

Revision ID: 116c8ccbd06a
Revises: 
Create Date: 2026-11-08 17:35:50.273662

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '116c8ccbd06a'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # races
    op.create_table(
        'races',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('round_number', sa.Integer(), nullable=False),
        sa.Column('race_name', sa.String(), nullable=False),
        sa.Column('circuit_type', sa.String(), nullable=True),
        sa.Column('event_date', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('year', 'round_number', name='uq_race_year_round')
    )

    # drivers
    op.create_table(
        'drivers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('code', sa.String(length=3), nullable=False),
        sa.Column('full_name', sa.String(), nullable=False),
        sa.Column('team_name', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code')
    )

    # model_versions
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('version_tag', sa.String(), nullable=False),
        sa.Column('training_date', sa.String(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('precision', sa.Float(), nullable=True),
        sa.Column('recall', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('mean_podiums', sa.Float(), nullable=True),
        sa.Column('perfect_podiums', sa.Integer(), nullable=True),
        sa.Column('races_evaluated', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('version_tag')
    )

    # predictions
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('race_id', sa.Integer(), nullable=False),
        sa.Column('driver_id', sa.Integer(), nullable=False),
        sa.Column('model_version_id', sa.Integer(), nullable=False),
        sa.Column('stage', sa.String(), nullable=False),
        sa.Column('podium_probability', sa.Float(), nullable=False),
        sa.Column('predicted_rank', sa.Integer(), nullable=False),
        sa.Column('predicted_podium', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['driver_id'], ['drivers.id']),
        sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id']),
        sa.ForeignKeyConstraint(['race_id'], ['races.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('race_id', 'driver_id', 'stage', name='uq_prediction_race_driver_stage')
    )

    # api_keys
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key_hash', sa.String(), nullable=False),
        sa.Column('client_name', sa.String(), nullable=False),
        sa.Column('tier', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key_hash')
    )

    # api_usage
    op.create_table(
        'api_usage',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('api_key_id', sa.Integer(), nullable=True),
        sa.Column('client_ip', sa.String(), nullable=False),
        sa.Column('endpoint', sa.String(), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id']),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('api_usage')
    op.drop_table('api_keys')
    op.drop_table('predictions')
    op.drop_table('model_versions')
    op.drop_table('drivers')
    op.drop_table('races')
