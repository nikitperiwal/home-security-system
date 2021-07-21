from plyer.utils import platform
from plyer import notification


def create_notification(title, message):
    """ Creates a Native OS Notification """

    notification.notify(
        title=title,
        message=message,
        app_name='Home Security Server',
        app_icon='resources/icon.' + ('ico' if platform == 'win' else 'png')
    )
