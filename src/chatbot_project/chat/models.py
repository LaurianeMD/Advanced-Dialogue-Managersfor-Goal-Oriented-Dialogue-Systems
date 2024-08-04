from django.db import models

class Conversation(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user_message[:50]} - {self.bot_response[:50]}"
