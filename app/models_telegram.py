"""ORM model for Telegram chats (migration 0004)."""

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class TelegramChat(Base):
    __tablename__ = "telegram_chats"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[str | None] = mapped_column(Text)
    alert_query: Mapped[str | None] = mapped_column(Text)
    alerts_on: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    last_alert_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
