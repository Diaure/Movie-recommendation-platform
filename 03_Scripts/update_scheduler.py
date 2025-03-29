import time
import subprocess

def update_movies():
    """Exécute le script de mise à jour des films."""
    print("Mise à jour des films en cours...")
    subprocess.run(["python", "../BD_A_IGNORE/traitement.py"])
    print("Mise à jour terminée.")

def schedule_update():
    """Lance la mise à jour tous les jours à une heure précise."""
    while True:
        current_time = time.strftime("%H:%M")  # Récupère l'heure actuelle (format HH:MM)
        if current_time == "03:00":  # Déclenche la mise à jour à 03h00
            update_movies()
            time.sleep(86400)  # Attendre 24 heures avant la prochaine mise à jour
        time.sleep(60)  # Vérifie l'heure chaque minute

if __name__ == "__main__":
    schedule_update()