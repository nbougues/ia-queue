# Installer un créneau IA sur un autre Hermes

Le service qui décide (« le serveur IA est-il libre ? ») est **unique** :
`http://inference-host:8765`, posé sur le serveur IA. Tout le reste est local à
chaque agent. Deux cas de figure.

---

## Cas A — un autre profil Hermes sur **cette** machine (agent-host)

Déjà fait pour `default` (House) et `finance` (Finance). Pour un
profil `<PROFIL>` supplémentaire :

1. **Le déclarer dans `config.json`** (`agents.<PROFIL>`) :

   ```json
   "<PROFIL>": {
     "binary": "<PROFIL>",                       // hermes si profil default
     "label": "Nom lisible",
     "enabled": true,
     "queue_path": "Hermes-<Nom>/AI Queue",
     "runlog_path": "Hermes-<Nom>/Run Log",
     "wf_home": "~/workflowy-<nom>",
     "check_free": true
   }
   ```

2. **Créer son identité Workflowy** (sinon il écrirait dans la racine d'un autre) :

   ```bash
   mkdir -p ~/workflowy-<nom>
   # config.json à l'intérieur : agent, root, write_roots = ["Hermes-<Nom>", "inbox"],
   # default_parent = "Hermes-<Nom>/Inbox"   (copier workflowy-finance/config.json)
   WF_HOME=~/workflowy-<nom> python3 ~/workflowy/wf.py whoami
   WF_HOME=~/workflowy-<nom> python3 ~/workflowy/wf.py \
       mkdir "Hermes-<Nom>/AI Queue" "Hermes-<Nom>/AI Queue/In Progress" \
             "Hermes-<Nom>/AI Queue/Done"
   ```

   ⚠️ L'API Workflowy est limitée à **1 écriture/minute par compte** (quota partagé
   entre tous les `WF_HOME`) : espacer les créations, sinon `export rate limited`.

3. **Wrapper + cron** :

   ```bash
   cd ~/ia-queue && ./install-agents.sh <PROFIL>
   ```

4. **Vérifier** :

   ```bash
   python3 ~/ia-queue/ia_slot_gate.py --agent <PROFIL> --dry-run
   <BINAIRE> cron list
   ```

> `check_free: false` si le modèle de ce profil ne tourne pas sur le GPU de .32
> (cas de `ops`, qui appelle un modèle distant) : le créneau ne se déclencherait
> jamais utilement.

---

## Cas B — un Hermes sur **une autre machine**

### Sur le serveur IA (inference-host) — une seule fois, déjà fait

```bash
# vérification depuis la nouvelle machine
curl -s -m 5 http://inference-host:8765/free | python3 -m json.tool
```

Doit répondre un JSON avec `free` et `reason`. Si ça ne répond pas : le service
`ia-free` est mort (`sudo systemctl restart ia-free` sur .32) ou la machine n'est
pas sur le réseau privé → **stop**, rien à installer.

### Sur la nouvelle machine

1. **Récupérer le code** (même chemin conseillée) :

   ```bash
   rsync -a agent-host:~/ia-queue/ ~/ia-queue/
   ```

2. **Un client Workflowy**. `wf.py` vit dans `~/workflowy/` avec sa clé
   dans `~/workdir/.workflowy_api_key`. Soit copier ce dossier (même compte),
   soit pointer `config.json → wf` vers un client équivalent respectant la même
   CLI (`todo list`, `mv`, `todo done`, `capture`, `note get`, `mkdir`).

3. **Adapter `config.json`** sur cette machine : `wf`, `workdir`, et l'entrée
   `agents.<PROFIL>` de la machine (nouvelle `label`, `queue_path`, `wf_home`).
   **Ne pas toucher** aux entrées des autres machines si le fichier est partagé.

4. **Le bail global reste le point de coordination** : la nouvelle machine n'a pas
   besoin d'être joignable, elle sort en HTTP vers .32. Vérifier que le port 8765
   est joignable depuis elle (firewall).

5. **Wrapper + cron** :

   ```bash
   cd ~/ia-queue && ./install-agents.sh <PROFIL>
   python3 ia_slot_gate.py --agent <PROFIL> --dry-run
   ```

6. **Tester une fois de bout en bout** :

   ```bash
   python3 ~/workflowy/wf.py todo add "EXEMPLE — test créneau <machine>" \
       --path "<racine>/AI Queue"
   python3 ~/ia-queue/ia_slot_worker.py --agent <PROFIL> --no-jitter
   tail -1 ~/ia-queue/logs/runs.jsonl | python3 -m json.tool
   <BINAIRE> sessions list | grep "AI-QUEUE"
   ```

   Attendu : `outcome: done`, `pinned_early: true`, une session
   `AI-QUEUE test créneau <machine>` épinglée, la tâche passée dans `Done`, une
   ligne dans `Run Log`.

---

## Ce qu'il ne faut PAS faire

- Installer **deux** services `ia-free` : le bail n'aurait plus de sens.
- Laisser deux profils partager la même file **et** la même identité Workflowy sans
  le bail : doubles claims.
- Mettre le `sleep` du jitter dans le script cron : le scheduler attend la fin du
  script et les ticks se chevaucheraient.
- Utiliser `--yolo` : le worker tourne en **mode smart** (les commandes dangereuses
  sont refusées, la tâche est non prioritaire, donc on contourne plutôt qu'on force).

## Rollback

```bash
<BINAIRE> cron pause <job_id>                                  # suspend un agent
# ou, dans config.json : "agents": {"<PROFIL>": {"enabled": false}}
sudo systemctl stop disable ia-free                            # sur .32, tout arrêter
```
